#!/usr/bin/env python3
"""Long-term value stock screener — self-contained, minimal network dependency.

Strategy:
- Use baostock for A-share financial data (TCP protocol, usually accessible)
- Per-query timeout via signal (Unix) to prevent hangs
- Process stocks sequentially with robust error handling
"""

import json
import os
import sys
import time
import signal
import math
from contextlib import contextmanager
from datetime import datetime

import pandas as pd
import baostock as bs
from policy_alpha import fetch_industry, fetch_growth_data, get_policy_score, score_policy_dimension
from institutional_holdings import fetch_institutional, score_institutional, get_institutional_summary
from industry_cycle import assess_cycle, score_cycle_dimension, get_cycle_summary, get_seasonal_phase
from controller_risk import analyze_controller, score_controller_dimension, get_controller_summary

# ── Timeout context manager ────────────────────────────────────────────

@contextmanager
def timeout(seconds: int):
    """Raise TimeoutError if block takes too long (Unix only)."""
    def _handler(signum, frame):
        raise TimeoutError(f"Query timed out after {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ── Data fetching ──────────────────────────────────────────────────────

def fetch_universe():
    """Get A-share stock list."""
    bs.login()
    try:
        rs = bs.query_stock_basic()
        df = rs.get_data()
        if df is None or df.empty:
            return pd.DataFrame(columns=["code", "code_name"])
        if "type" in df.columns:
            df = df[df["type"] == "1"].copy()
        return df
    finally:
        bs.logout()


from typing import Optional, Dict, Any

# ... later ...

def fetch_one_stock(code: str, name: str) -> Optional[Dict[str, Any]]:
    """Fetch financial metrics for one stock. Must be called within active baostock session.
    Returns None on timeout or fatal error."""
    m = {"code": code.replace("sh.", "SH").replace("sz.", "SZ").replace("bj.", "BJ"), "name": name}
    try:
        _fetch_latest(bs, code, m)
        _fetch_history(bs, code, m)
        _fetch_balance(bs, code, m)
        _fetch_policy(bs, code, m)
    except TimeoutError:
        return None
    except Exception as e:
        err = str(e)
        if "接收数据异常" in err or "timeout" in err.lower():
            return None
        raise

    _fetch_institutional(code, m)
    _fetch_cycle(m)
    _fetch_controller(code, m)
    _set_defaults(m)
    return m


def _fetch_latest(bs_mod, code, m):
    """Use Q4 (annual) data for reliable ratios. Q1 may have empty revenue."""
    for (y, q) in [(2025, 4), (2024, 4), (2023, 4), (2025, 1)]:
        rs = bs_mod.query_profit_data(code=code, year=y, quarter=q)
        if rs.error_code != "0":
            continue
        df = rs.get_data()
        if df.empty:
            continue
        r = df.iloc[-1]
        rev = _f(r, "MBRevenue")
        if rev > 0:
            m["revenue_latest"] = round(rev / 1e8, 1)
        m["roe_latest"] = round(_f(r, "roeAvg") * 100, 1)
        m["net_margin"] = round(_f(r, "npMargin") * 100, 1)
        m["gross_margin"] = round(_f(r, "gpMargin") * 100, 1)
        np_val = _f(r, "netProfit")
        if np_val > 0:
            m["net_profit_latest"] = round(np_val / 1e8, 1)
        return


def _fetch_history(bs_mod, code, m):
    h = {"rev": [], "profit": [], "roe": [], "gm": []}
    for year in [2021, 2022, 2023, 2024, 2025]:
        rs = bs_mod.query_profit_data(code=code, year=year, quarter=4)
        if rs.error_code != "0":
            continue
        df = rs.get_data()
        if df.empty:
            continue
        r = df.iloc[-1]
        h["rev"].append(_f(r, "MBRevenue"))
        h["profit"].append(_f(r, "netProfit"))
        h["roe"].append(_f(r, "roeAvg") * 100)
        h["gm"].append(_f(r, "gpMargin") * 100)

    revs = pd.Series(h["rev"]).dropna()
    profits = pd.Series(h["profit"]).dropna()
    roes = pd.Series(h["roe"]).dropna()
    gms = pd.Series(h["gm"]).dropna()

    if len(revs) >= 3 and revs.iloc[-3] > 0:
        m["revenue_cagr_3y"] = round(((revs.iloc[-1] / revs.iloc[-3]) ** (1 / 3) - 1) * 100, 1)
        m["revenue_up_years"] = min(sum(1 for i in range(1, len(revs)) if revs.iloc[i] > revs.iloc[i - 1]), 3)
    if len(revs) >= 2 and revs.iloc[-2] > 0:
        m["revenue_growth"] = round((revs.iloc[-1] / revs.iloc[-2] - 1) * 100, 1)

    if len(profits) >= 3 and profits.iloc[-3] > 0:
        r = profits.iloc[-1] / profits.iloc[-3]
        if r > 0:
            m["profit_cagr_3y"] = round((r ** (1 / 3) - 1) * 100, 1)
        m["profit_up_years"] = min(sum(1 for i in range(1, len(profits)) if profits.iloc[i] > profits.iloc[i - 1]), 3)

    if len(profits) >= 4 and profits.iloc[-2] > 0 and profits.iloc[-3] > 0:
        growth_latest = (profits.iloc[-1] / profits.iloc[-2] - 1) * 100
        growth_prior = (profits.iloc[-2] / profits.iloc[-3] - 1) * 100
        m["earnings_accel"] = round(growth_latest - growth_prior, 1)
        m["earnings_growth_latest"] = round(growth_latest, 1)
        m["earnings_growth_prior"] = round(growth_prior, 1)

    if len(roes) >= 3:
        m["roe_5y_min"] = round(roes.min(), 1)
        m["roe_min_15"] = m["roe_5y_min"] >= 15
        m["roe_min_10"] = m["roe_5y_min"] >= 10

    if len(gms) >= 3:
        m["gross_margin_stable"] = (gms.max() - gms.min()) < 10

    consecutive = 0
    for p in reversed(profits):
        if p > 0: consecutive += 1
        else: break
    m["consecutive_profit_years"] = consecutive


def _fetch_balance(bs_mod, code, m):
    for (y, q) in [(2025, 4), (2024, 4), (2023, 4)]:
        rs = bs_mod.query_balance_data(code=code, year=y, quarter=q)
        if rs.error_code != "0":
            continue
        df = rs.get_data()
        if df.empty:
            continue
        r = df.iloc[-1]
        m["current_ratio"] = round(_f(r, "currentRatio"), 2)
        # assetToEquity: debt_ratio% = (1 - 1/ATE) * 100
        ate = _f(r, "assetToEquity")
        if ate > 1:
            m["debt_ratio"] = round((1 - 1 / ate) * 100, 1)
        return


def _fetch_policy(bs_mod, code, m):
    """Fetch industry classification and YoY growth for policy scoring."""
    industry = fetch_industry(bs_mod, code)
    m["industry"] = industry or ""
    pname, pscore = get_policy_score(industry or "")
    m["policy_name"] = pname or ""
    m["policy_score"] = pscore
    growth = fetch_growth_data(bs_mod, code)
    m["yoy_ni"] = growth.get("yoy_ni", 0)
    m["yoy_eps"] = growth.get("yoy_eps", 0)


def _fetch_institutional(code, m):
    """Fetch institutional holdings data (akshare). Runs outside baostock session."""
    result = fetch_institutional(code)
    m["inst_result"] = result
    m["inst_huijin"] = result["huijin"] if result else False
    m["inst_shebao"] = result["shebao"] if result else False
    m["inst_xianzi"] = result["xianzi"] if result else False
    m["inst_northbound"] = result["northbound"] if result else False
    m["inst_qfii"] = result["qfii"] if result else False
    m["inst_categories"] = result["num_categories"] if result else 0
    m["inst_total_pct"] = result["total_inst_pct"] if result else 0
    m["inst_summary"] = get_institutional_summary(result)
    m["inst_changes"] = result.get("changes", {}) if result else {}


def _fetch_cycle(m):
    """Assess industry cycle and seasonality from existing financial data."""
    cycle = assess_cycle(m)
    m["cycle_result"] = cycle
    m["cycle_stage"] = cycle["cycle_stage"]
    m["cycle_outlook"] = cycle["outlook"]
    m["cycle_risk_level"] = cycle["risk_level"]
    m["cycle_risk_flags"] = cycle["risk_flags"]
    m["cycle_summary"] = get_cycle_summary(cycle)
    ind = m.get("industry", "")
    phase, note = get_seasonal_phase(ind)
    m["seasonal_phase"] = phase
    m["seasonal_note"] = note


def _fetch_controller(code, m):
    """Analyze controller type, region, and moral hazard."""
    inst_result = m.get("inst_result")
    ctrl = analyze_controller(code, inst_result)
    m["ctrl_result"] = ctrl
    m["ctrl_type"] = ctrl.get("controller_type", "未知")
    m["ctrl_region"] = ctrl.get("region_group", "未知")
    m["ctrl_north_south"] = ctrl.get("north_south", "未知")
    m["ctrl_province"] = ctrl.get("province", "")
    m["ctrl_moral_risk"] = ctrl.get("moral_risk_level", "无")
    m["ctrl_moral_flags"] = ctrl.get("moral_flags", [])
    m["ctrl_summary"] = get_controller_summary(ctrl)


def _set_defaults(m):
    m.setdefault("roe_latest", 0); m.setdefault("roe_5y_min", 0)
    m.setdefault("roe_min_15", False); m.setdefault("roe_min_10", False)
    m.setdefault("net_margin", 0); m.setdefault("gross_margin", 0)
    m.setdefault("gross_margin_stable", False); m.setdefault("roic", 0)
    m.setdefault("debt_ratio", 50); m.setdefault("current_ratio", 1.0)
    m.setdefault("consecutive_profit_years", 0); m.setdefault("fcf_to_net_profit", 0)
    m.setdefault("revenue_cagr_3y", 0); m.setdefault("profit_cagr_3y", 0)
    m.setdefault("earnings_accel", 0); m.setdefault("earnings_growth_latest", 0)
    m.setdefault("earnings_growth_prior", 0)
    m.setdefault("revenue_up_years", 0); m.setdefault("profit_up_years", 0)
    m.setdefault("growth_accelerating", False); m.setdefault("revenue_growth", 0)
    m.setdefault("net_profit_latest", 0); m.setdefault("revenue_latest", 0)
    m.setdefault("pe_ttm", 0); m.setdefault("pb", 0)
    m.setdefault("pe_quantile_5y", 50); m.setdefault("pb_quantile_5y", 50)
    m.setdefault("pe_pb_product", 999); m.setdefault("industry_pe", 30)
    m.setdefault("industry", ""); m.setdefault("policy_name", ""); m.setdefault("policy_score", 0)
    m.setdefault("yoy_ni", 0); m.setdefault("yoy_eps", 0)
    m.setdefault("inst_result", None); m.setdefault("inst_huijin", False)
    m.setdefault("inst_shebao", False); m.setdefault("inst_xianzi", False)
    m.setdefault("inst_northbound", False); m.setdefault("inst_qfii", False)
    m.setdefault("inst_categories", 0); m.setdefault("inst_total_pct", 0)
    m.setdefault("inst_summary", ""); m.setdefault("inst_changes", {})
    m.setdefault("cycle_result", {}); m.setdefault("cycle_stage", "—")
    m.setdefault("cycle_outlook", "—"); m.setdefault("cycle_risk_level", "—")
    m.setdefault("cycle_risk_flags", []); m.setdefault("cycle_summary", "—")
    m.setdefault("seasonal_phase", "—"); m.setdefault("seasonal_note", "")
    m.setdefault("ctrl_result", {}); m.setdefault("ctrl_type", "未知")
    m.setdefault("ctrl_region", "未知"); m.setdefault("ctrl_north_south", "未知")
    m.setdefault("ctrl_province", ""); m.setdefault("ctrl_moral_risk", "无")
    m.setdefault("ctrl_moral_flags", []); m.setdefault("ctrl_summary", "未知")
    m.setdefault("div_yield", 0); m.setdefault("dividend_years", 0)
    m.setdefault("industry_leader", False); m.setdefault("net_margin_above_median", False)
    m.setdefault("mcap_yi", 0); m.setdefault("price", 0)


def _f(row, key, default=0):
    try:
        v = row.get(key)
        return float(v) if v is not None and v != "" else default
    except (ValueError, TypeError):
        return default


# ── Scoring engine ─────────────────────────────────────────────────────

def score_stock(m: dict) -> dict:
    """9-dimension scoring with industry-relative adjustment.

    Returns dict with scores, metrics, and industry_z (z-score quality).
    """
    def dim(label, weight, calc_fn):
        raw = max(1, min(10, calc_fn(m)))
        return {"label": label, "score": raw, "weight": weight}

    dims = [
        dim("盈利能力", 0.20, _score_profitability),
        dim("成长性", 0.16, _score_growth),
        dim("财务健康", 0.15, _score_health),
        dim("估值", 0.13, _score_valuation),
        dim("护城河", 0.08, _score_moat),
        dim("政策风口", 0.08, _score_policy),
        dim("机构持仓", 0.08, _score_institutional),
        dim("行业周期", 0.07, _score_cycle),
        dim("实控风险", 0.05, _score_controller),
    ]

    # ── Industry-relative z-score adjustment ──
    from industry_benchmark import score_with_industry_benchmarks, z_to_score
    ib = score_with_industry_benchmarks(m)
    z_quality = ib["industry_quality_score"]  # 0-10

    # Blend profitability: 70% absolute + 30% industry-relative
    # Profitability z-score combines ROE and margin z-scores
    z_profit = (z_to_score(ib["z_roe"]) + z_to_score(ib["z_margin"])) / 2
    dims[0]["score"] = round(dims[0]["score"] * 0.70 + z_profit * 0.30, 1)

    # Blend growth: 70% absolute + 30% industry-relative
    z_growth = z_to_score(ib["z_growth"])
    dims[1]["score"] = round(dims[1]["score"] * 0.70 + z_growth * 0.30, 1)

    # Blend health: debt z-score adjusts by up to ±1
    z_debt = z_to_score(ib["z_debt"])
    dims[2]["score"] = round(max(1, min(10, dims[2]["score"] + (z_debt - 5) * 0.2)), 1)

    total = round(sum(d["score"] * d["weight"] for d in dims) * 10, 1)
    return {
        "code": m["code"], "name": m["name"], "total": total,
        "profitability": round(dims[0]["score"], 1),
        "growth": round(dims[1]["score"], 1),
        "health": round(dims[2]["score"], 1),
        "valuation": round(dims[3]["score"], 1),
        "moat": round(dims[4]["score"], 1),
        "policy": round(dims[5]["score"], 1),
        "institutional": round(dims[6]["score"], 1),
        "cycle": round(dims[7]["score"], 1),
        "controller": round(dims[8]["score"], 1),
        "industry_z": round(z_quality, 1),
        "metrics": {"roe": m["roe_latest"], "pe": m["pe_ttm"], "debt": m["debt_ratio"],
                     "div_yield": m["div_yield"], "mcap_yi": m["mcap_yi"],
                     "rev_cagr": round(m["revenue_cagr_3y"], 1),
                     "profit_cagr": round(m["profit_cagr_3y"], 1),
                     "policy_name": m.get("policy_name", ""),
                     "policy_score": m.get("policy_score", 0),
                     "inst_summary": m.get("inst_summary", ""),
                     "inst_huijin": m.get("inst_huijin", False),
                     "inst_shebao": m.get("inst_shebao", False),
                     "inst_xianzi": m.get("inst_xianzi", False),
                     "inst_northbound": m.get("inst_northbound", False),
                     "inst_total_pct": m.get("inst_total_pct", 0),
                     "cycle_stage": m.get("cycle_stage", "—"),
                     "cycle_outlook": m.get("cycle_outlook", "—"),
                     "cycle_risk_flags": m.get("cycle_risk_flags", []),
                     "seasonal_phase": m.get("seasonal_phase", "—"),
                     "ctrl_type": m.get("ctrl_type", "—"),
                     "ctrl_north_south": m.get("ctrl_north_south", "—"),
                     "ctrl_moral_risk": m.get("ctrl_moral_risk", "—"),
                     "ctrl_moral_flags": m.get("ctrl_moral_flags", [])},
    }


def _score_profitability(m):
    s = 5; roe = m.get("roe_latest", 0) or 0
    if roe >= 20: s += 4
    elif roe >= 15: s += 3
    elif roe >= 10: s += 2
    elif roe < 5: s -= 2
    rm = m.get("roe_5y_min", 0) or 0
    if rm >= 15: s += 3
    elif rm >= 12: s += 2
    nm = m.get("net_margin", 0) or 0
    if nm >= 20: s += 2
    elif nm >= 15: s += 1
    return s


def _score_growth(m):
    s = 5; r = m.get("revenue_cagr_3y", 0) or 0
    if r >= 25: s += 4
    elif r >= 20: s += 3
    elif r >= 15: s += 2
    elif r < 0: s -= 2
    p = m.get("profit_cagr_3y", 0) or 0
    if p >= 20: s += 3
    elif p >= 15: s += 2
    elif p < 0: s -= 1
    if m.get("revenue_up_years", 0) >= 3: s += 1
    if m.get("profit_up_years", 0) >= 3: s += 1
    # Earnings acceleration: 2nd derivative of profit growth
    accel = m.get("earnings_accel", 0) or 0
    if accel > 15: s += 2       # strong acceleration — inflection point
    elif accel > 5: s += 1      # moderate acceleration
    elif accel < -15: s -= 2    # sharp deceleration
    elif accel < -5: s -= 1     # moderate deceleration
    return s


def _score_health(m):
    s = 5; d = m.get("debt_ratio", 50) or 50
    if d < 30: s += 3
    elif d < 40: s += 2
    elif d > 60: s -= 2
    cr = m.get("current_ratio", 1) or 1
    if cr > 2: s += 1
    elif cr > 1.5: s += 0.5
    if m.get("consecutive_profit_years", 0) >= 5: s += 2
    elif m.get("consecutive_profit_years", 0) >= 3: s += 1
    if m.get("fcf_to_net_profit", 0) > 0.8: s += 2
    dy = m.get("dividend_years", 0) or 0
    if dy >= 5: s += 2
    elif dy >= 3: s += 1
    return s


def _score_valuation(m):
    s = 5; pq = m.get("pe_quantile_5y", 50) or 50
    if pq < 30: s += 3
    elif pq < 50: s += 2
    elif pq >= 85: s -= 2
    bq = m.get("pb_quantile_5y", 50) or 50
    if bq < 30: s += 2
    elif bq < 50: s += 1
    elif bq >= 85: s -= 1
    pe, pb = m.get("pe_ttm", 0) or 0, m.get("pb", 0) or 0
    if pe and pb and pe < 15 and pb < 1.5: s += 1
    if pe and pe < (m.get("industry_pe", 30) or 30): s += 1
    if pe and pb and pe * pb < 22.5: s += 1
    dy = m.get("div_yield", 0) or 0
    if dy >= 3: s += 2
    elif dy >= 2: s += 1
    return s


def _score_moat(m):
    s = 5; gm = m.get("gross_margin", 0) or 0
    if gm >= 40: s += 1
    if m.get("gross_margin_stable"): s += 1
    if m.get("roe_min_15"): s += 2
    elif m.get("roe_min_10"): s += 1
    if m.get("industry_leader"): s += 1
    if m.get("net_margin_above_median"): s += 1
    if (m.get("dividend_years", 0) or 0) >= 5: s += 2
    return s


def _score_policy(m):
    """Score policy dimension using policy_alpha module."""
    return score_policy_dimension(
        m.get("policy_name", ""),
        m.get("policy_score", 0),
        {"yoy_ni": m.get("yoy_ni", 0), "yoy_eps": m.get("yoy_eps", 0)},
    )


def _score_institutional(m):
    """Score institutional holdings dimension."""
    result = m.get("inst_result")
    return score_institutional(result)


def _score_cycle(m):
    """Score industry cycle dimension."""
    cycle = m.get("cycle_result", {})
    if not cycle:
        return 5
    return score_cycle_dimension(cycle)


def _score_controller(m):
    """Score controller & moral hazard dimension."""
    ctrl = m.get("ctrl_result", {})
    if not ctrl:
        return 5
    return score_controller_dimension(ctrl)


# ── Filters ────────────────────────────────────────────────────────────

ST_PREFIXES = ("ST", "*ST", "N", "C", "PT", "NST")

def filter_basic(code: str, name: str) -> bool:
    """Tier 1: basic stock filter."""
    if code.startswith("sh.000") or code.startswith("sz.399"):
        return False  # indices
    if any(name.startswith(p) for p in ST_PREFIXES):
        return False  # ST
    if not code.startswith(("sh.6", "sz.0", "sz.3", "bj.")):
        return False  # B-share, NEEQ
    return True


def filter_financial(m: Dict[str, Any]) -> Optional[str]:
    """Tier 2: financial criteria. Returns fail reason or None."""
    roe = m.get("roe_latest") or 0
    np_val = m.get("net_profit_latest") or 0
    rev = m.get("revenue_latest") or 0
    debt = m.get("debt_ratio") or 100

    # Detect stocks with no financial data (delisted/historical)
    if roe == 0 and rev == 0 and np_val == 0:
        return "数据不可用"

    if roe < 8:
        return f"ROE={roe:.1f}%"
    if np_val <= 0:
        return "净亏损"
    if rev < 3:
        return f"营收={rev:.1f}亿"
    if debt > 70:
        return f"负债率={debt:.1f}%"
    return None


# ── Main pipeline ──────────────────────────────────────────────────────

def run_screen(max_stocks: int = 0, top_n: int = 30, rate_limit: float = 10.0):
    """Run full screen pipeline. Returns (rankings, stats)."""
    t0 = time.monotonic()
    print("Fetching A-share universe...", file=sys.stderr, flush=True)
    df = fetch_universe()

    # Tier 1
    candidates = []
    for _, row in df.iterrows():
        code = str(row.iloc[0]).strip()
        name = str(row.iloc[1]).strip()
        if filter_basic(code, name):
            candidates.append((code, name))

    stats = {"universe": len(df), "t1_passed": len(candidates)}
    print(f"  {len(candidates)} stocks passed basic filter", file=sys.stderr, flush=True)

    if max_stocks > 0:
        candidates = candidates[:max_stocks]
        print(f"  (capped at {max_stocks})", file=sys.stderr, flush=True)

    # Tier 2 + Scoring — single baostock session for all stocks
    bs.login()
    try:
        results = []
        failures = 0
        no_data_count = 0
        for i, (code, name) in enumerate(candidates):
            if (i + 1) % 50 == 0:
                e = time.monotonic() - t0
                rate = (i + 1) / e if e > 0 else 0
                print(f"  {i+1}/{len(candidates)} ({rate:.1f}/s) | {len(results)} passed",
                      file=sys.stderr, flush=True)

            time.sleep(1.0 / rate_limit)

            m = fetch_one_stock(code, name)
            if m is None:
                failures += 1
                continue

            fail = filter_financial(m)
            if fail:
                if fail == "数据不可用":
                    no_data_count += 1
                continue

            results.append(score_stock(m))
    finally:
        bs.logout()

    results.sort(key=lambda r: r["total"], reverse=True)
    results = results[:top_n]

    stats.update({
        "elapsed": round(time.monotonic() - t0, 1),
        "t2_passed": len(results),
        "failures": failures,
        "no_data": no_data_count,
    })
    return results, stats


# ── Output ─────────────────────────────────────────────────────────────

GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"; BOLD = "\033[1m"; RESET = "\033[0m"

def _c(v):
    if v >= 8: return f"{GREEN}{v:.1f}{RESET}"
    if v >= 5: return f"{YELLOW}{v:.1f}{RESET}"
    return f"{RED}{v:.1f}{RESET}"

def print_table(results):
    n = len(results)
    h = f"{'#':>3}  {'代码':<8}  {'名称':<10}  {'总分':>5}  {'盈利':>4}  {'成长':>4}  {'健康':>4}  {'估值':>4}  {'护城河':>4}"
    print(f"\n{BOLD}🏆 TOP {n} 长期投资选股排名{RESET}")
    print("-" * 80)
    print(h)
    print("-" * 80)
    for i, r in enumerate(results):
        row = (f"{i+1:>3}  {r['code']:<8}  {r['name']:<10}  "
               f"{BOLD}{r['total']:>5.1f}{RESET}  "
               f"{_c(r['profitability']):>12}  {_c(r['growth']):>12}  "
               f"{_c(r['health']):>12}  {_c(r['valuation']):>12}  {_c(r['moat']):>12}")
        print(row)
    print("-" * 80)

def print_stats(stats):
    print(f"\n📊 流水线: {stats['universe']} 只 → T1: {stats['t1_passed']} → T2: {stats['t2_passed']} | {stats['elapsed']}s")

def save_json(results, stats, path=None):
    path = path or os.path.join(os.path.dirname(__file__), ".cache", "screen_universe", "result.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"ts": datetime.now().isoformat(), "stats": stats, "rankings": results}, f, ensure_ascii=False, indent=2)
    return path
