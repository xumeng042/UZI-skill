#!/usr/bin/env python3
"""价值投资筛选模块 — 高股息+央国企+基本面好+估值低.

对所有通过基础筛选的股票进行价值评分，按总分排名。
不做硬性过滤，让分数说话。

评分维度:
- 股息回报(30%): 股息率越高越好，连续分红年数加分
- 估值安全边际(25%): PE/PB越低越好
- 基本面质量(25%): ROE、净利率、负债率
- 稳定性(20%): 连续盈利年数、分红年数、毛利率稳定
"""

import sys
import os
import time
from typing import Optional, Dict, List, Tuple

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, os.path.join(SCRIPTS_DIR, "lib"))

from screen_stocks import (
    fetch_universe, filter_basic, filter_financial,
    fetch_one_stock, score_stock,
)


# ── 央国企识别 ──────────────────────────────────────────────────────────

def is_soe(m: dict) -> bool:
    """Check if stock is a central or local state-owned enterprise."""
    ctrl = m.get("ctrl_type", "") or ""
    return any(kw in ctrl for kw in ["央企", "地方国企", "国资"])


def soe_label(m: dict) -> str:
    """Get SOE category label."""
    ctrl = m.get("ctrl_type", "") or ""
    if "央企" in ctrl: return "央企"
    if "地方国企" in ctrl: return "地方国企"
    if "国资" in ctrl: return "国资"
    return "非国企"


# ── 价值投资评分 ────────────────────────────────────────────────────────

def score_value_stock(m: dict) -> dict:
    """Score a stock for value investing (0-100 scale)."""

    # ── 1. 股息回报 (30分) ──
    dy = m.get("div_yield", 0) or 0
    div_years = m.get("dividend_years", 0) or 0

    div_score = 0
    if dy >= 7:   div_score = 28
    elif dy >= 6: div_score = 24
    elif dy >= 5: div_score = 20
    elif dy >= 4: div_score = 16
    elif dy >= 3: div_score = 10
    elif dy >= 2: div_score = 5
    else:         div_score = 0

    # 连续分红加分
    if div_years >= 10: div_score += 4
    elif div_years >= 5: div_score += 2
    elif div_years >= 3: div_score += 1

    div_score = min(30, div_score)

    # ── 2. 估值安全边际 (25分) ──
    pe_q = m.get("pe_quantile_5y", 50) or 50
    pb_q = m.get("pb_quantile_5y", 50) or 50
    pe = m.get("pe_ttm", 0) or 0
    pb = m.get("pb", 0) or 0

    val_score = 0
    # PE分位越低越好
    if pe_q <= 10:     val_score += 10
    elif pe_q <= 20:   val_score += 8
    elif pe_q <= 30:   val_score += 6
    elif pe_q <= 40:   val_score += 4
    elif pe_q <= 50:   val_score += 2
    # PB分位越低越好
    if pb_q <= 10:     val_score += 8
    elif pb_q <= 20:   val_score += 6
    elif pb_q <= 30:   val_score += 5
    elif pb_q <= 40:   val_score += 3
    elif pb_q <= 50:   val_score += 1
    # 绝对估值也低（深度价值）
    if pe and pe < 10 and pb and pb < 1.0: val_score += 5
    elif pe and pe < 15 and pb and pb < 1.5: val_score += 2

    val_score = min(25, val_score)

    # ── 3. 基本面质量 (25分) ──
    roe = m.get("roe_latest", 0) or 0
    nm = m.get("net_margin", 0) or 0
    debt = m.get("debt_ratio", 50) or 50

    fund_score = 0
    if roe >= 20:          fund_score += 8
    elif roe >= 15:        fund_score += 6
    elif roe >= 12:        fund_score += 5
    elif roe >= 10:        fund_score += 3

    if nm >= 20:           fund_score += 6
    elif nm >= 15:         fund_score += 4
    elif nm >= 10:         fund_score += 2

    if debt < 20:          fund_score += 6
    elif debt < 30:        fund_score += 5
    elif debt < 40:        fund_score += 3
    elif debt < 50:        fund_score += 1

    # ROE稳定性
    if m.get("roe_min_15"): fund_score += 5
    elif m.get("roe_min_10"): fund_score += 3

    fund_score = min(25, fund_score)

    # ── 4. 稳定性 (20分) ──
    profit_years = m.get("consecutive_profit_years", 0) or 0
    rev_up = m.get("revenue_up_years", 0) or 0
    gm_stable = m.get("gross_margin_stable", False)

    stab_score = 0
    if profit_years >= 10:   stab_score += 7
    elif profit_years >= 5:  stab_score += 5
    elif profit_years >= 3:  stab_score += 3
    elif profit_years >= 1:  stab_score += 1

    if rev_up >= 3:          stab_score += 4
    elif rev_up >= 2:        stab_score += 2

    if gm_stable:            stab_score += 3

    # 连续分红
    if div_years >= 10:      stab_score += 6
    elif div_years >= 5:     stab_score += 4
    elif div_years >= 3:     stab_score += 2

    stab_score = min(20, stab_score)

    total = div_score + val_score + fund_score + stab_score

    return {
        "code": m.get("code", ""),
        "name": m.get("name", ""),
        "total": total,
        "div_score": div_score,
        "val_score": val_score,
        "fund_score": fund_score,
        "stab_score": stab_score,
        "div_yield": dy,
        "div_years": div_years,
        "pe_q": pe_q,
        "pb_q": pb_q,
        "roe": roe,
        "debt": debt,
        "soe": soe_label(m),
        "is_soe": is_soe(m),
    }


# ── 主筛选管道 ──────────────────────────────────────────────────────────

def run_value_screening(max_stocks: int = 0, top_n: int = 20,
                         rate_limit: float = 5.0) -> Tuple[List[Dict], Dict]:
    """Run value investing screening pipeline.

    Scores all candidates on value metrics and ranks by total score.
    No hard filters — every stock gets a score, top N win.
    """
    import baostock as bs

    t0 = time.monotonic()
    print("Fetching A-share universe for value investing...", file=sys.stderr, flush=True)
    df = fetch_universe()

    candidates = []
    for _, row in df.iterrows():
        code = str(row.iloc[0]).strip()
        name = str(row.iloc[1]).strip()
        if filter_basic(code, name):
            candidates.append((code, name))

    stats = {"universe": len(df), "t1_passed": len(candidates)}

    if max_stocks > 0:
        candidates = candidates[:max_stocks]

    results = []
    failures = 0

    bs.login()
    try:
        for i, (code, name) in enumerate(candidates):
            if (i + 1) % 50 == 0:
                elapsed = time.monotonic() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  {i+1}/{len(candidates)} ({rate:.1f}/s) | {len(results)} scored",
                      file=sys.stderr, flush=True)

            time.sleep(1.0 / rate_limit)

            m = fetch_one_stock(code, name)
            if m is None:
                failures += 1
                continue

            # Score every stock — let the score do the ranking, not hard filters
            scored = score_value_stock(m)
            results.append(scored)
    finally:
        bs.logout()

    results.sort(key=lambda r: r["total"], reverse=True)
    results = results[:top_n]

    elapsed = time.monotonic() - t0
    stats.update({
        "elapsed": round(elapsed, 1),
        "passed": len(results),
        "failures": failures,
    })

    return results, stats


# ── CLI test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="价值投资选股")
    p.add_argument("--max", type=int, default=300, help="max stocks to scan")
    p.add_argument("--top", type=int, default=20, help="top N output")
    p.add_argument("--rate", type=float, default=5.0, help="queries/sec")
    args = p.parse_args()

    results, stats = run_value_screening(
        max_stocks=args.max, top_n=args.top, rate_limit=args.rate,
    )

    print(f"\n{'='*90}")
    print(f"  💰 价值投资组合 (股息>4% · 央国企 · 基本面好 · 估值低)")
    print(f"{'='*90}")
    print(f"{'#':>3} {'代码':<9} {'名称':<8} {'总分':>5} {'股息':>5} {'分红年':>4} "
          f"{'PE%':>5} {'PB%':>5} {'ROE':>5} {'负债':>5} {'类型':<6} "
          f"{'股息':>4} {'估值':>4} {'质量':>4} {'稳定':>4}")
    print("-" * 90)

    for i, r in enumerate(results):
        print(f"{i+1:>3} {r['code']:<9} {r['name']:<8} {r['total']:>5.1f} "
              f"{r['div_yield']:>4.1f}% {r['div_years']:>4} "
              f"{r['pe_q']:>4.0f}% {r['pb_q']:>4.0f}% {r['roe']:>4.1f}% {r['debt']:>4.1f}% "
              f"{r['soe']:<6} {r['div_score']:>4.0f} {r['val_score']:>4.0f} "
              f"{r['fund_score']:>4.0f} {r['stab_score']:>4.0f}")

    print(f"\nPipeline: {stats['universe']} → T1:{stats['t1_passed']} → Passed:{stats['passed']} | {stats['elapsed']}s")
