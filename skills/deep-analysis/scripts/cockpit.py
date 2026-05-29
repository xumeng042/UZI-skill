#!/usr/bin/env python3
"""驾驶舱数据模块 — 实时市场全景数据采集.

Sections:
1. 指数快照 — 上证/深证/创业板/科创50
2. 市场宽度 — 涨停数、成交额、上证PE、换手率
3. 风险预警 — 跌停股、异常波动
4. 市场要闻 — 全市场重要新闻(带链接)
5. 北向资金 — 沪股通+深股通净流向
"""

import time
import math
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from statistics import mean

import pandas as pd

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)


# ── Helpers ────────────────────────────────────────────────────────────────

def _safe_call(fn, default=None, label="", retries=1):
    """Wrap a callable with try/except + retry, return default on failure."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt < retries:
                wait = (attempt + 1) * 3  # 3s backoff
                if label:
                    print(f"  [cockpit] {label} retry {attempt+1}/{retries} in {wait}s: {e}",
                          file=sys.stderr, flush=True)
                time.sleep(wait)
    if label:
        print(f"  [cockpit] {label} FAILED: {last_err}", file=sys.stderr, flush=True)
    return default


def _sparkline(values, width=14, color_up="#26a69a", color_down="#ef5350"):
    """Render mini sparkline as HTML spans from a list of floats."""
    if not values or len(values) < 2:
        return ""
    mn, mx = min(values), max(values)
    if mx <= mn:
        mx = mn + 1
    bars = "▁▂▃▄▅▆▇█"
    result = []
    for v in values[-width:]:
        idx = int((v - mn) / (mx - mn) * (len(bars) - 1))
        idx = max(0, min(len(bars) - 1, idx))
        result.append(f'<span style="font-size:0.85em;">{bars[idx]}</span>')
    return "".join(result)


def _mini_change_badge(value, digits=2):
    """Inline HTML: green up / red down / grey flat."""
    if value is None:
        return '<span style="color:#8b949e;">--</span>'
    if value > 0:
        return f'<span style="color:#26a69a;font-weight:600;">+{value:.{digits}f}%</span>'
    elif value < 0:
        return f'<span style="color:#ef5350;font-weight:600;">{value:.{digits}f}%</span>'
    else:
        return f'<span style="color:#8b949e;">{value:.{digits}f}%</span>'


# ── 1. 指数快照 ────────────────────────────────────────────────────────────

_INDEX_CODES = {
    "上证指数": "sh.000001",
    "深证成指": "sz.399001",
    "创业板指": "sz.399006",
    "科创50": "sh.588000",  # 科创50ETF as proxy (baostock lacks sh.000688)
}


def fetch_index_snapshot() -> Dict:
    """Fetch 4 major indices with 60-day sparkline data via baostock."""
    result = {"indices": [], "_error": None}

    try:
        import baostock as bs
        bs.login()
        try:
            for name, code in _INDEX_CODES.items():
                try:
                    rs = bs.query_history_k_data_plus(
                        code, "date,close", start_date="2025-01-01",
                        end_date=datetime.now().strftime("%Y-%m-%d"),
                        frequency="d", adjustflag="2",
                    )
                    df = rs.get_data() if rs.error_code == "0" else None
                    if df is None or df.empty:
                        result["indices"].append({
                            "name": name, "code": code, "price": 0,
                            "change_pct": 0, "sparkline": [],
                            "_error": "无数据",
                        })
                        continue

                    closes = [float(c) for c in df["close"]]
                    last = closes[-1]
                    prev = closes[-2] if len(closes) >= 2 else last
                    change_pct = (last / prev - 1) * 100 if prev else 0

                    result["indices"].append({
                        "name": name,
                        "code": code,
                        "price": round(last, 2),
                        "change_pct": round(change_pct, 2),
                        "sparkline": closes[-60:] if len(closes) >= 10 else closes,
                        "_error": None,
                    })
                except Exception:
                    result["indices"].append({
                        "name": name, "code": code, "price": 0,
                        "change_pct": 0, "sparkline": [], "_error": "获取失败",
                    })
        finally:
            bs.logout()
    except Exception as e:
        result["_error"] = str(e)[:80]
        # Fallback: try akshare for at least latest prices
        try:
            import akshare as ak
            df = _safe_call(lambda: ak.stock_zh_index_daily_em(symbol="sh000001"), label="akshare index")
            if df is not None and not df.empty:
                last_close = float(df["close"].iloc[-1])
                result["indices"] = [
                    {"name": n, "code": c, "price": last_close if n == "上证指数" else 0,
                     "change_pct": 0, "sparkline": [], "_error": None if n == "上证指数" else "无数据"}
                    for n, c in _INDEX_CODES.items()
                ]
        except Exception:
            pass

    return result


# ── 2. 市场宽度 ────────────────────────────────────────────────────────────

def fetch_market_breadth() -> Dict:
    """Market breadth from SSE daily deal + limit-up pools.

    Uses stock_sse_deal_daily (SSE summary) and stock_zt_pool_em (limit-up).
    Falls back gracefully when East Money push2 API is unreachable.
    """
    result = {"advance": 0, "decline": 0, "flat": 0,
              "limit_up": 0, "limit_down": 0, "volume_yi": 0,
              "total_stocks": 0, "sse_pe": 0, "sse_turnover": 0,
              "_error": None}

    try:
        import akshare as ak

        # SSE daily summary — works reliably (not push2 API)
        sse = _safe_call(lambda: ak.stock_sse_deal_daily(), label="sse_deal")
        if sse is not None and not sse.empty:
            try:
                vol_row = sse[sse["单日情况"] == "成交金额"]
                if not vol_row.empty:
                    result["volume_yi"] = round(float(vol_row["股票"].iloc[0]), 0)
                pe_row = sse[sse["单日情况"] == "平均市盈率"]
                if not pe_row.empty:
                    result["sse_pe"] = round(float(pe_row["股票"].iloc[0]), 1)
                to_row = sse[sse["单日情况"] == "换手率"]
                if not to_row.empty:
                    result["sse_turnover"] = round(float(to_row["股票"].iloc[0]), 2)
                count_row = sse[sse["单日情况"] == "挂牌数"]
                if not count_row.empty:
                    result["total_stocks"] = int(float(count_row["股票"].iloc[0]))
            except Exception:
                pass

        # Limit-up pools (these work — not push2 API)
        today = datetime.now().strftime("%Y%m%d")
        zt = _safe_call(lambda: ak.stock_zt_pool_em(date=today), label="zt_pool")
        if zt is not None and not zt.empty:
            result["limit_up"] = len(zt)

        # Strong stocks pool for advance/decline approximation
        strong = _safe_call(lambda: ak.stock_zt_pool_strong_em(date=today), label="zt_strong")
        if strong is not None and not strong.empty:
            # These are stocks near limit-up, use as advance proxy
            result["advance"] = len(strong)

        # Try A-share spot for full breadth (may fail)
        spot = _safe_call(
            lambda: ak.stock_zh_a_spot_em(),
            label="stock_zh_a_spot", retries=0,
        )
        if spot is not None and not spot.empty:
            changes = pd.to_numeric(spot["涨跌幅"], errors="coerce")
            volumes = pd.to_numeric(spot["成交额"], errors="coerce")
            result["advance"] = int((changes > 0).sum())
            result["decline"] = int((changes < 0).sum())
            result["flat"] = int((changes == 0).sum())
            result["volume_yi"] = round(volumes.sum() / 1e8, 0)
            result["total_stocks"] = len(spot)
            result["limit_down"] = int((changes < -9.5).sum())

    except Exception as e:
        result["_error"] = str(e)[:80]

    return result

    return result


# ── 3. 板块热力图 ──────────────────────────────────────────────────────────

def fetch_risk_alerts() -> Dict:
    """Identify risk sectors and abnormal stocks."""
    result = {"alerts": [], "abnormal_stocks": [], "_error": None}

    try:
        import akshare as ak

        # Risk sectors: >3% decline
        board = _safe_call(lambda: ak.stock_board_industry_name_em(), label="risk_board", retries=0)
        if board is not None and not board.empty:
            for _, row in board.iterrows():
                chg = float(row.get("涨跌幅", 0) or 0)
                if chg < -3:
                    result["alerts"].append({
                        "type": "板块风险",
                        "description": f"{row.get('板块名称', '')} 板块跌幅 {chg:.2f}%",
                        "severity": "high" if chg < -5 else "medium",
                    })

        # Near-limit-down stocks from spot data
        spot = _safe_call(lambda: ak.stock_zh_a_spot_em(), label="risk_spot", retries=0)
        if spot is not None and not spot.empty:
            for _, row in spot.iterrows():
                chg = float(row.get("涨跌幅", 0) or 0)
                if chg < -9.5:
                    result["alerts"].append({
                        "type": "个股风险",
                        "description": f"{row.get('名称', '')}({row.get('代码', '')}) 跌 {chg:.2f}%，接近跌停",
                        "severity": "high",
                    })
                    result["abnormal_stocks"].append({
                        "code": str(row.get("代码", "")),
                        "name": str(row.get("名称", "")),
                        "change_pct": round(chg, 2),
                        "reason": "跌超9.5%",
                    })

        if not result["alerts"]:
            result["alerts"] = []

    except Exception as e:
        result["_error"] = str(e)[:80]

    return result


# ── 6. 市场要闻 ────────────────────────────────────────────────────────────

def fetch_market_news(limit: int = 8) -> Dict:
    """Fetch recent market-wide news with URLs."""
    result = {"news": [], "_error": None}

    try:
        import akshare as ak
        df = _safe_call(lambda: ak.stock_info_global_em(), label="market_news")
        if df is None or df.empty:
            # Try alternative: sina global
            df = _safe_call(lambda: ak.stock_info_global_sina(), label="market_news_sina")

        if df is None or df.empty:
            result["_error"] = "新闻数据不可用"
            return result

        title_col = next((c for c in ["标题", "title", "内容"] if c in df.columns), df.columns[0])
        time_col = next((c for c in ["发布时间", "time", "pub_time"] if c in df.columns), None)
        url_col = next((c for c in ["链接", "url", "link"] if c in df.columns), None)
        summary_col = next((c for c in ["摘要", "summary", "desc"] if c in df.columns), None)

        for _, row in df.head(limit).iterrows():
            item = {
                "title": str(row.get(title_col, ""))[:100],
                "time": str(row.get(time_col, ""))[:19] if time_col else "",
            }
            if url_col:
                item["url"] = str(row.get(url_col, ""))
            if summary_col:
                item["summary"] = str(row.get(summary_col, ""))[:120]
            result["news"].append(item)

    except Exception as e:
        result["_error"] = str(e)[:80]

    return result


# ── 7. 北向资金 ────────────────────────────────────────────────────────────

def fetch_northbound_flow(days: int = 20) -> Dict:
    """Fetch north-bound capital flow (沪股通+深股通 net flow)."""
    result = {"latest_net_yi": 0, "cumulative_5d_yi": 0, "cumulative_20d_yi": 0,
              "daily_flows": [], "direction": "未知", "_error": None}

    try:
        import akshare as ak

        # Use correct symbol "北向资金" (not "北上")
        nb = _safe_call(
            lambda: ak.stock_hsgt_hist_em(symbol="北向资金"),
            label="northbound",
        )

        if nb is not None and not nb.empty:
            # Find the flow column (akshare returns "当日成交净买额")
            flow_col = next((c for c in ["当日成交净买额", "净流入", "资金流向"] if c in nb.columns), None)
            date_col = next((c for c in ["日期", "date"] if c in nb.columns), None)

            if flow_col:
                flows = []
                for _, row in nb.iterrows():
                    try:
                        val = float(row[flow_col])
                        if math.isnan(val):
                            continue
                        flows.append({
                            "date": str(row.get(date_col, ""))[:10] if date_col else "",
                            "net_yi": round(val / 1e8, 2),
                        })
                    except (ValueError, TypeError):
                        continue

                if flows:
                    result["daily_flows"] = flows[-days:]
                    # Use last valid (non-NaN) value for latest
                    valid_flows = [f for f in result["daily_flows"] if not (math.isnan(f["net_yi"]) if isinstance(f["net_yi"], float) else False)]
                    if valid_flows:
                        result["latest_net_yi"] = valid_flows[-1]["net_yi"]
                        result["cumulative_5d_yi"] = round(sum(f["net_yi"] for f in valid_flows[-5:]), 2)
                        result["cumulative_20d_yi"] = round(sum(f["net_yi"] for f in valid_flows[-20:]), 2)
                        result["direction"] = "流入" if result["latest_net_yi"] > 0 else "流出"
            else:
                # Try separate 沪股通/深股通
                sh = _safe_call(lambda: ak.stock_em_hsgt_north_net_flow_in(indicator="沪股通"), label="nb_sh")
                sz = _safe_call(lambda: ak.stock_em_hsgt_north_net_flow_in(indicator="深股通"), label="nb_sz")
                if sh is not None and sz is not None:
                    try:
                        sh_val = float(sh.iloc[-1, 1]) if len(sh.columns) >= 2 else 0
                        sz_val = float(sz.iloc[-1, 1]) if len(sz.columns) >= 2 else 0
                        result["latest_net_yi"] = round((sh_val + sz_val) / 1e8, 2)
                        result["direction"] = "流入" if result["latest_net_yi"] > 0 else "流出"
                    except Exception:
                        pass
        else:
            result["_error"] = "北向资金数据不可用"

    except Exception as e:
        result["_error"] = str(e)[:80]

    return result


# ── 编排器 ─────────────────────────────────────────────────────────────────

def get_cockpit_data(use_threads: bool = True) -> Dict[str, Any]:
    """Orchestrate all 5 data fetches. Returns dict keyed by section name.

    Each section dict has data fields + optional '_error' key.
    The orchestrator never raises — individual failures are captured in _error.
    """
    t0 = time.monotonic()
    sections = {}

    fetchers = {
        "index_snapshot": fetch_index_snapshot,
        "market_breadth": fetch_market_breadth,
        "risk_alerts": fetch_risk_alerts,
        "market_news": fetch_market_news,
        "northbound_flow": fetch_northbound_flow,
    }

    # Sequential execution to avoid rate-limiting from East Money APIs
    for key, fn in fetchers.items():
        print(f"  [cockpit] fetching {key}...", file=sys.stderr, flush=True)
        try:
            sections[key] = fn()
        except Exception as e:
            sections[key] = {"_error": str(e)[:80]}

    sections["fetched_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = time.monotonic() - t0
    print(f"  [cockpit] all fetches done in {elapsed:.1f}s", file=sys.stderr, flush=True)

    return sections


# ── CLI test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    print("Fetching cockpit data...", flush=True)
    data = get_cockpit_data(use_threads=True)

    for key, section in data.items():
        if key == "fetched_at":
            continue
        if section.get("_error"):
            print(f"  ✗ {key}: {section['_error']}")
        else:
            print(f"  ✓ {key}: OK")

    print(f"\nFetched at: {data.get('fetched_at', 'N/A')}")

    # Summary
    idx = data.get("index_snapshot", {}).get("indices", [])
    print("\nIndices:")
    for i in idx:
        print(f"  {i['name']}: {i['price']} ({i['change_pct']:+.2f}%) {'[ERR]' if i.get('_error') else ''}")

    breadth = data.get("market_breadth", {})
    print(f"\nBreadth: ↑{breadth.get('advance',0)} ↓{breadth.get('decline',0)} "
          f"—{breadth.get('flat',0)}  涨停{breadth.get('limit_up',0)} 跌停{breadth.get('limit_down',0)}")

    heat = data.get("sector_heatmap", {})
    print(f"Sector heatmap: {len(heat.get('top_gainers',[]))} gainers, {len(heat.get('top_losers',[]))} losers")
    for s in heat.get("top_gainers", [])[:3]:
        print(f"  + {s['name']}: {s['change_pct']:+.2f}%")

    nb = data.get("northbound_flow", {})
    print(f"Northbound: {nb.get('latest_net_yi',0):+.2f}亿 ({nb.get('direction','?')})")
