#!/usr/bin/env python3
"""量化模式 — 实时涨跌概率 + 新闻舆情 + 资金动向.

Components:
1. 技术面(40%): 日内量价、短期动量、波动率
2. 资金面(30%): 主力资金净流向、北向资金
3. 舆情面(30%): 近期新闻关键词情感分析

Output: 1日/3日/5日上涨概率
"""

import sys
import os
import time
import math
from statistics import mean
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, os.path.join(SCRIPTS_DIR, "lib"))

from trend_predict import fetch_klines, compute_timeframe_indicators, score_timeframe


# ── 新闻舆情分析 ────────────────────────────────────────────────────────

# 关键词情感词典
_POSITIVE_KW = [
    "业绩增长", "净利润增长", "营收增长", "大幅增长", "超预期",
    "中标", "签约", "突破", "创新高", "获批", "上市",
    "回购", "增持", "分红", "高送转", "股权激励",
    "利好", "重大合同", "战略合作", "扩产", "产能释放",
    "订单", "交付", "量产", "技术突破", "专利",
    "机构买入", "北向资金增持", "主力资金流入",
]

_NEGATIVE_KW = [
    "业绩下滑", "净利润下降", "营收下降", "亏损", "预亏",
    "减持", "套现", "质押", "爆雷", "违约",
    "处罚", "罚款", "立案", "调查", "退市",
    "风险提示", "异常波动", "停牌", "终止",
    "下滑", "下降", "减少", "萎缩", "低迷",
    "商誉减值", "计提", "坏账", "诉讼",
    "大股东减持", "高管减持", "限售解禁",
]


def _analyze_news_sentiment(news_df) -> Dict:
    """Analyze news sentiment from akshare stock_news_em output.

    Returns dict with sentiment score, positive_count, negative_count, headlines.
    """
    if news_df is None or news_df.empty:
        return {"score": 0, "positive": 0, "negative": 0,
                "total": 0, "headlines": [], "label": "无新闻"}

    pos_count = 0
    neg_count = 0
    headlines = []

    for _, row in news_df.iterrows():
        title = str(row.get("title", "") or row.get("标题", "") or "")
        if not title:
            continue

        # Check positive keywords
        for kw in _POSITIVE_KW:
            if kw in title:
                pos_count += 1
                if len(headlines) < 8:
                    headlines.append({"title": title, "sentiment": "positive"})
                break
        else:
            # Check negative keywords
            for kw in _NEGATIVE_KW:
                if kw in title:
                    neg_count += 1
                    if len(headlines) < 8:
                        headlines.append({"title": title, "sentiment": "negative"})
                    break

    total = pos_count + neg_count
    # Sentiment score: -10 to +10
    if total > 0:
        score = (pos_count - neg_count) / total * 10
    else:
        score = 0

    if score > 3:
        label = "偏多"
    elif score > 0:
        label = "中性偏多"
    elif score < -3:
        label = "偏空"
    elif score < 0:
        label = "中性偏空"
    else:
        label = "中性"

    return {
        "score": round(score, 1),
        "positive": pos_count,
        "negative": neg_count,
        "total": total,
        "headlines": headlines,
        "label": label,
    }


def fetch_stock_news(code: str) -> Optional[Dict]:
    """Fetch recent news for a stock via akshare."""
    try:
        import akshare as ak
        # Convert baostock code to akshare format
        # sh.600519 → 600519, sz.000858 → 000858
        symbol = code.replace("sh.", "").replace("sz.", "").replace("bj.", "")
        news = ak.stock_news_em(stock=symbol)
        if news is not None and not news.empty:
            # Only keep last 7 days
            recent = news.head(30)
            return _analyze_news_sentiment(recent)
    except Exception:
        pass
    return None


# ── 资金流向分析 ────────────────────────────────────────────────────────

def fetch_fund_flow(code: str) -> Optional[Dict]:
    """Fetch recent capital flow for a stock via akshare."""
    try:
        import akshare as ak
        symbol = code.replace("sh.", "").replace("sz.", "").replace("bj.", "")
        market = "sh" if code.startswith("sh") else "sz"

        flow = ak.stock_individual_fund_flow(stock=symbol, market=market)
        if flow is None or flow.empty:
            return None

        recent = flow.head(5)  # last 5 days
        main_net = [float(r) for r in recent.get("主力净流入-净额", []) if r]
        total_net = [float(r) for r in recent.get("总净流入-净额", []) if r]

        if not main_net:
            return None

        # Score based on recent main force flow
        main_5d = sum(main_net)  # net over 5 days
        main_1d = main_net[0] if main_net else 0

        # Normalize by some estimate of market cap (rough)
        # Score: -10 to +10
        if main_5d > 5000:
            flow_score = 8
        elif main_5d > 2000:
            flow_score = 6
        elif main_5d > 500:
            flow_score = 4
        elif main_5d > 0:
            flow_score = 2
        elif main_5d > -500:
            flow_score = -1
        elif main_5d > -2000:
            flow_score = -3
        elif main_5d > -5000:
            flow_score = -5
        else:
            flow_score = -7

        # Check for consecutive inflow/outflow days
        pos_days = sum(1 for v in main_net if v > 0)
        neg_days = sum(1 for v in main_net if v < 0)

        if pos_days >= 4:
            signal = "持续流入"
        elif pos_days >= 3:
            signal = "偏流入"
        elif neg_days >= 4:
            signal = "持续流出"
        elif neg_days >= 3:
            signal = "偏流出"
        else:
            signal = "震荡"

        return {
            "main_1d": round(main_1d, 0),
            "main_5d": round(main_5d, 0),
            "score": round(flow_score, 1),
            "signal": signal,
            "pos_days": pos_days,
            "neg_days": neg_days,
        }
    except Exception:
        return None


# ── 量化综合评分 ────────────────────────────────────────────────────────

def score_quant_stock(bs, code: str, name: str,
                       news: Optional[Dict] = None,
                       flow: Optional[Dict] = None) -> Optional[Dict]:
    """Generate quant trading probability for a single stock.

    Returns dict with 1D/3D/5D probabilities and factor breakdown.
    """
    # ── Technical analysis ──
    daily = fetch_klines(bs, code, "d", start="2025-01-01")
    if not daily["closes"] or len(daily["closes"]) < 20:
        return None

    closes = daily["closes"]
    highs = daily["highs"]
    lows = daily["lows"]
    vols = daily["volumes"]

    last = closes[-1]

    # 1. Short-term momentum (5-day ROC)
    roc_5d = (closes[-1] / closes[-min(5, len(closes))] - 1) * 100 if len(closes) >= 5 else 0
    roc_10d = (closes[-1] / closes[-min(10, len(closes))] - 1) * 100 if len(closes) >= 10 else 0

    # 2. Volume surge
    avg_vol_5 = mean(vols[-5:]) if len(vols) >= 5 else 1
    avg_vol_20 = mean(vols[-20:]) if len(vols) >= 20 else 1
    vol_ratio = avg_vol_5 / avg_vol_20 if avg_vol_20 > 0 else 1

    # 3. Intra-range position
    n = min(20, len(closes))
    range_high = max(highs[-n:])
    range_low = min(lows[-n:])
    range_pos = (last - range_low) / (range_high - range_low) * 100 if range_high > range_low else 50

    # 4. Bollinger %B
    ma20 = mean(closes[-20:]) if len(closes) >= 20 else last
    std20 = (sum((c - ma20)**2 for c in closes[-20:]) / 20) ** 0.5 if len(closes) >= 20 else 1
    bb_pct = (last - (ma20 - 2*std20)) / (4*std20) * 100 if std20 > 0 else 50

    # 5. RSI
    from trend_predict import _rsi
    rsi = _rsi(closes, 7)  # shorter RSI for quant mode

    # ── Technical score (0-50) ──
    tech_score = 25  # neutral

    # Momentum
    if roc_5d > 5:     tech_score += 8
    elif roc_5d > 2:   tech_score += 4
    elif roc_5d > 0:   tech_score += 1
    elif roc_5d < -5:  tech_score -= 8
    elif roc_5d < -2:  tech_score -= 4

    # Volume
    if vol_ratio > 1.5 and roc_5d > 0:
        tech_score += 5  # 放量上涨
    elif vol_ratio > 1.5 and roc_5d < 0:
        tech_score -= 5  # 放量下跌
    elif vol_ratio < 0.5:
        tech_score -= 3  # 缩量

    # RSI
    if 30 <= rsi <= 65:
        tech_score += 4  # healthy zone
    elif 20 <= rsi < 30:
        tech_score += 2  # oversold — potential bounce
    elif rsi > 80:
        tech_score -= 3  # overbought — potential pullback

    # Range position
    if 30 <= range_pos <= 70:
        tech_score += 3  # room to run
    elif range_pos < 20:
        tech_score -= 2  # near support (could bounce or break)
    elif range_pos > 85:
        tech_score -= 3  # near resistance

    tech_score = max(0, min(50, tech_score))

    # ── News sentiment score (0-30) ──
    if news:
        news_score = max(0, min(30, (news["score"] + 10) * 1.5))  # -10..+10 → 0..30
    else:
        news_score = 15  # neutral when no data

    # ── Capital flow score (0-20) ──
    if flow:
        flow_score = max(0, min(20, (flow["score"] + 10)))  # -10..+10 → 0..20
    else:
        flow_score = 10  # neutral when no data

    # ── Combined probability ──
    total_score = tech_score + news_score + flow_score  # 0-100
    prob_1d = min(95, max(5, total_score * 0.7 + 15))  # map toward 50
    prob_3d = min(95, max(5, total_score * 0.8 + 10))
    prob_5d = min(95, max(5, total_score * 0.85 + 7.5))

    return {
        "code": code.replace("sh.", "SH").replace("sz.", "SZ").replace("bj.", "BJ"),
        "name": name,
        "prob_1d": round(prob_1d, 1),
        "prob_3d": round(prob_3d, 1),
        "prob_5d": round(prob_5d, 1),
        "total_score": round(total_score, 1),
        "tech_score": round(tech_score, 1),
        "news_score": round(news_score, 1),
        "flow_score": round(flow_score, 1),
        "roc_5d": round(roc_5d, 2),
        "vol_ratio": round(vol_ratio, 2),
        "rsi_7d": round(rsi, 1),
        "range_pos": round(range_pos, 1),
        "news_label": news["label"] if news else "无数据",
        "news_headlines": news["headlines"] if news else [],
        "flow_signal": flow["signal"] if flow else "无数据",
        "flow_5d_net": flow["main_5d"] if flow else 0,
    }


# ── 批量量化扫描 ────────────────────────────────────────────────────────

def run_quant_scan(max_stocks: int = 80, top_n: int = 15,
                    rate_limit: float = 3.0,
                    with_news: bool = True,
                    with_flow: bool = True) -> Tuple[List[Dict], Dict]:
    """Run quant mode batch scan.

    Args:
        max_stocks: max stocks to scan
        top_n: top N to return
        rate_limit: queries per second
        with_news: fetch news sentiment
        with_flow: fetch capital flow data

    Returns (results, stats).
    """
    import baostock as bs
    from screen_stocks import fetch_universe, filter_basic

    t0 = time.monotonic()

    df = fetch_universe()
    candidates = []
    for _, row in df.iterrows():
        code = str(row.iloc[0]).strip()
        name = str(row.iloc[1]).strip()
        if filter_basic(code, name):
            candidates.append((code, name))

    if max_stocks > 0:
        candidates = candidates[:max_stocks]

    stats = {"universe": len(df), "t1_passed": len(candidates)}

    results = []
    failures = 0
    news_hits = 0
    flow_hits = 0

    bs.login()
    try:
        for i, (code, name) in enumerate(candidates):
            if (i + 1) % 20 == 0:
                elapsed = time.monotonic() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  {i+1}/{len(candidates)} ({rate:.1f}/s) | {len(results)} quanted",
                      file=sys.stderr, flush=True)

            time.sleep(1.0 / rate_limit)

            # Fetch news (a bit slower, skip some)
            news = None
            if with_news and i % 3 == 0:  # every 3rd stock to save time
                news = fetch_stock_news(code)
                if news:
                    news_hits += 1

            # Fetch capital flow
            flow = None
            if with_flow:
                flow = fetch_fund_flow(code)
                if flow:
                    flow_hits += 1

            # Score
            scored = score_quant_stock(bs, code, name, news, flow)
            if scored is None:
                failures += 1
                continue

            results.append(scored)
    finally:
        bs.logout()

    results.sort(key=lambda r: r["total_score"], reverse=True)
    results = results[:top_n]

    elapsed = time.monotonic() - t0
    stats.update({
        "elapsed": round(elapsed, 1),
        "quant_passed": len(results),
        "failures": failures,
        "news_hits": news_hits,
        "flow_hits": flow_hits,
    })

    return results, stats


# ── CLI test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="量化模式 — 实时涨跌概率")
    p.add_argument("--max", type=int, default=50, help="max stocks to scan")
    p.add_argument("--top", type=int, default=15, help="top N output")
    p.add_argument("--rate", type=float, default=3.0, help="queries/sec")
    p.add_argument("--no-news", action="store_true", help="skip news")
    p.add_argument("--no-flow", action="store_true", help="skip flow")
    args = p.parse_args()

    results, stats = run_quant_scan(
        max_stocks=args.max, top_n=args.top,
        rate_limit=args.rate,
        with_news=not args.no_news,
        with_flow=not args.no_flow,
    )

    print(f"\n{'='*100}")
    print(f"  📈 量化扫描 — 实时涨跌概率排名")
    print(f"{'='*100}")
    print(f"{'#':>3} {'代码':<9} {'名称':<8} {'1日':>5} {'3日':>5} {'5日':>5} "
          f"{'综合':>5} {'技术':>5} {'舆情':>5} {'资金':>5} {'5D动':>6} {'量比':>5} {'RSI':>5} "
          f"{'舆情':<8} {'资金':<8}")
    print("-" * 100)

    for i, r in enumerate(results):
        print(f"{i+1:>3} {r['code']:<9} {r['name']:<8} "
              f"{r['prob_1d']:>4.0f}% {r['prob_3d']:>4.0f}% {r['prob_5d']:>4.0f}% "
              f"{r['total_score']:>5.0f} {r['tech_score']:>5.0f} {r['news_score']:>5.0f} {r['flow_score']:>5.0f} "
              f"{r['roc_5d']:>+5.1f}% {r['vol_ratio']:>4.2f} {r['rsi_7d']:>5.0f} "
              f"{r['news_label']:<8} {r['flow_signal']:<8}")

    print(f"\nPipeline: {stats['universe']} → T1:{stats['t1_passed']} → Quant passed:{stats['quant_passed']} | {stats['elapsed']}s")
    if stats.get("news_hits"):
        print(f"  News coverage: {stats['news_hits']} | Flow coverage: {stats.get('flow_hits', 0)}")
