#!/usr/bin/env python3
"""Backtest the enhanced multi-timeframe trend prediction model.

Runs predictions on historical dates using the full enhanced pipeline:
- Market regime detection (with historical index data)
- Industry z-score normalization
- MACD/RSI divergence detection
- Probability calibration

Compares predicted upside probabilities against actual subsequent returns.
"""

import sys
import os
import time
import json
from datetime import datetime, timedelta
from statistics import mean, median, stdev
from typing import Optional, Dict, List, Tuple

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, os.path.join(SCRIPTS_DIR, "lib"))

from screen_stocks import fetch_universe, filter_basic, fetch_one_stock, score_stock, _f
from trend_predict import (
    fetch_klines, compute_timeframe_indicators, score_timeframe,
)
from market_regime import classify_regime, fetch_index_klines
from industry_benchmark import score_with_industry_benchmarks, compute_relative_momentum


# ── Enhanced prediction for historical dates ───────────────────────────

def predict_at_date_enhanced(bs, code: str, name: str, fin_data: dict,
                             end_date: str, regime: dict) -> Optional[Dict]:
    """Run enhanced prediction as of a specific historical date.

    Matches predict_one_stock() pipeline: divergence, industry z-score,
    horizon decay blending, earnings acceleration.
    No probability calibration (removed per backtest evidence).
    """
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    daily_start = (end_dt - timedelta(days=4*365)).strftime("%Y-%m-%d")
    weekly_start = (end_dt - timedelta(days=6*365)).strftime("%Y-%m-%d")
    monthly_start = (end_dt - timedelta(days=10*365)).strftime("%Y-%m-%d")

    daily = fetch_klines(bs, code, "d", start=daily_start, end=end_date)
    weekly = fetch_klines(bs, code, "w", start=weekly_start, end=end_date)
    monthly = fetch_klines(bs, code, "m", start=monthly_start, end=end_date)

    ind_d = compute_timeframe_indicators(daily)
    ind_w = compute_timeframe_indicators(weekly)
    ind_m = compute_timeframe_indicators(monthly)

    if ind_d is None:
        return None

    tech_d = score_timeframe(ind_d)
    tech_w = score_timeframe(ind_w) if ind_w else 50
    tech_m = score_timeframe(ind_m) if ind_m else 50

    # Financial score with industry z-score (from enhanced score_stock)
    if fin_data:
        scored = score_stock(fin_data)
        dim_scores = [
            scored["profitability"], scored["growth"], scored["health"],
            scored["valuation"], scored["moat"], scored["policy"],
            scored["institutional"], scored["cycle"], scored["controller"],
        ]
        fin_score = sum(dim_scores) / 9 * 10
        total = scored["total"]
    else:
        fin_score = 50
        total = 50
        scored = {}

    # Multi-timeframe resonance
    mtf_aligned = 0
    if ind_d and ind_w:
        if ind_d["above_ma20"] and ind_w["above_ma20"]:
            mtf_aligned += 1
        if ind_d["macd_above_zero"] and ind_w["macd_above_zero"]:
            mtf_aligned += 1
    if ind_w and ind_m:
        if ind_w["above_ma20"] and ind_m["above_ma20"]:
            mtf_aligned += 1

    resonance_bonus = mtf_aligned * 3

    # Raw probabilities (same blend as predict_one_stock)
    prob_1m = min(95, tech_d * 0.70 + tech_w * 0.30 + resonance_bonus * 0.5)
    prob_2m = min(95, tech_d * 0.35 + tech_w * 0.35 + fin_score * 0.30 + resonance_bonus * 0.5)
    prob_3m = min(95, tech_w * 0.50 + fin_score * 0.50 + resonance_bonus * 0.3)
    prob_6m = min(95, tech_m * 0.40 + fin_score * 0.60 + resonance_bonus * 0.3)

    # ── Horizon decay blending ──
    raw_1m = prob_1m
    raw_2m = prob_2m
    prob_1m = raw_1m * 0.25 + prob_3m * 0.40 + prob_6m * 0.35
    prob_2m = raw_2m * 0.30 + prob_3m * 0.35 + prob_6m * 0.35

    regime_score = regime.get("score", 50)

    # ADX for diagnostics
    adx_d = ind_d.get("adx", 20) or 20

    # ── Relative Strength (RS) alpha ──
    index_ret_1m = regime.get("ret_1m", 0) or 0
    index_ret_3m = regime.get("ret_3m", 0) or 0
    stock_ret_1m = ind_d.get("ret_1m", 0) or 0
    stock_ret_3m = ind_d.get("ret_3m", 0) or 0
    alpha_1m = stock_ret_1m - index_ret_1m
    alpha_3m = stock_ret_3m - index_ret_3m

    if alpha_1m > 10:       rs_bonus_1m = 2.0
    elif alpha_1m > 5:      rs_bonus_1m = 1.0
    elif alpha_1m < -10:    rs_bonus_1m = -2.0
    elif alpha_1m < -5:     rs_bonus_1m = -1.0
    else:                   rs_bonus_1m = 0

    if alpha_3m > 15:       rs_bonus_3m = 1.5
    elif alpha_3m > 8:      rs_bonus_3m = 0.8
    elif alpha_3m < -15:    rs_bonus_3m = -1.5
    elif alpha_3m < -8:     rs_bonus_3m = -0.8
    else:                   rs_bonus_3m = 0

    prob_1m = min(95, max(5, prob_1m + rs_bonus_1m))
    prob_2m = min(95, max(5, prob_2m + rs_bonus_1m * 0.7 + rs_bonus_3m * 0.3))
    prob_3m = min(95, max(5, prob_3m + rs_bonus_3m))
    prob_6m = min(95, max(5, prob_6m + rs_bonus_3m * 0.5))

    return {
        "code": code, "name": name,
        "prob_1m": prob_1m, "prob_2m": prob_2m,
        "prob_3m": prob_3m, "prob_6m": prob_6m,
        "raw_1m": round(raw_1m, 1), "raw_6m": round(raw_1m, 1),
        "fin_score": round(fin_score, 1), "total": total,
        "tech_d": tech_d, "tech_w": tech_w, "tech_m": tech_m,
        "mtf_resonance": mtf_aligned,
        "regime": regime.get("regime", "?"),
        "regime_score": regime_score,
        "div_macd_d": ind_d.get("macd_divergence", 0),
        "div_rsi_d": ind_d.get("rsi_divergence", 0),
        # For industry-relative momentum
        "ret_1m": round(stock_ret_1m, 1),
        "ret_3m": round(stock_ret_3m, 1),
        "industry": fin_data.get("industry", "") if fin_data else "",
        # RS alpha + ADX
        "alpha_1m": round(alpha_1m, 1),
        "alpha_3m": round(alpha_3m, 1),
        "adx_d": round(adx_d, 1),
    }


# ── Actual return measurement ──────────────────────────────────────────

def measure_actual_return(bs, code: str, start_date: str,
                          horizon_days: int) -> Optional[float]:
    """Get actual return from start_date to start_date + horizon_days."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = start_dt + timedelta(days=horizon_days + 5)

    klines = fetch_klines(bs, code, "d",
                          start=start_date,
                          end=end_dt.strftime("%Y-%m-%d"))
    if not klines["closes"] or len(klines["closes"]) < 2:
        return None

    start_price = klines["closes"][0]
    end_price = klines["closes"][-1]
    if start_price <= 0:
        return None
    return (end_price / start_price - 1) * 100


# ── Backtest engine ─────────────────────────────────────────────────────

def run_backtest(test_date: str, max_stocks: int = 150,
                 rate_limit: float = 5.0) -> Dict:
    """Run enhanced backtest for a single historical date."""
    import baostock as bs

    t0 = time.monotonic()
    print(f"\n{'='*60}")
    print(f"Enhanced Backtest: {test_date}")
    print(f"{'='*60}")

    # Fetch universe
    df = fetch_universe()
    candidates = []
    for _, row in df.iterrows():
        code = str(row.iloc[0]).strip()
        name = str(row.iloc[1]).strip()
        if filter_basic(code, name):
            candidates.append((code, name))

    if max_stocks > 0:
        candidates = candidates[:max_stocks]

    # ── Determine historical market regime ──
    bs.login()
    try:
        index_klines = fetch_index_klines(bs, "sh.000300", end_date=test_date)
        # Truncate to test_date
        if index_klines:
            regime = classify_regime(index_klines)
        else:
            regime = {"regime": "未知", "score": 50, "prob_multiplier": 1.0, "signals": []}
    finally:
        bs.logout()

    print(f"Market regime at {test_date}: {regime['regime']} (score={regime['score']}, mult={regime['prob_multiplier']})")
    print(f"Universe: {len(df)} -> Tier1: {len(candidates)} -> Testing: {min(max_stocks, len(candidates))}")

    results = []
    failures = 0

    bs.login()
    try:
        for i, (code, name) in enumerate(candidates):
            if (i + 1) % 20 == 0:
                elapsed = time.monotonic() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  {i+1}/{len(candidates)} ({rate:.1f}/s) | {len(results)} predicted",
                      file=sys.stderr, flush=True)

            time.sleep(1.0 / rate_limit)

            fin = fetch_one_stock(code, name)
            if fin is None:
                failures += 1
                continue

            # Bypass financial filter for unbiased backtest
            pred = predict_at_date_enhanced(bs, code, name, fin, test_date, regime)
            if pred is None:
                failures += 1
                continue

            # Measure actual subsequent returns
            horizons = {"1m": 21, "2m": 42, "3m": 63, "6m": 126}
            actual = {}
            for label, days in horizons.items():
                ret = measure_actual_return(bs, code, test_date, days)
                actual[f"ret_{label}"] = ret

            if actual["ret_1m"] is None:
                continue

            results.append({**pred, **actual})
    finally:
        bs.logout()

    elapsed = time.monotonic() - t0

    # ── Industry-relative momentum adjustment ──
    mom_map = compute_relative_momentum(results)
    for r in results:
        mom = mom_map.get(r["code"], {})
        adj = mom.get("combined", 0)
        r["prob_1m"] = round(min(95, max(5, r["prob_1m"] + adj * 1.0)), 1)
        r["prob_2m"] = round(min(95, max(5, r["prob_2m"] + adj * 0.7)), 1)
        r["prob_3m"] = round(min(95, max(5, r["prob_3m"] + adj * 0.3)), 1)
        r["mom_adj"] = adj
        r["mom_peers"] = mom.get("peer_count", 0)

    # ── Compute accuracy metrics per horizon ──
    metrics = {}
    for horizon, days in [("1m", 21), ("2m", 42), ("3m", 63), ("6m", 126)]:
        prob_key = f"prob_{horizon}"
        ret_key = f"ret_{horizon}"

        valid = [r for r in results if r.get(ret_key) is not None]
        if len(valid) < 5:
            metrics[horizon] = {"error": f"only {len(valid)} valid samples"}
            continue

        n = len(valid)

        # Directional accuracy
        correct = sum(
            1 for r in valid
            if (r[prob_key] >= 50 and (r[ret_key] or 0) > 0) or
               (r[prob_key] < 50 and (r[ret_key] or 0) <= 0)
        )
        direction_acc = correct / n * 100

        # Up-capture
        high_confidence = [r for r in valid if r[prob_key] >= 60]
        up_hit_rate = (sum(1 for r in high_confidence if (r[ret_key] or 0) > 0) /
                       len(high_confidence) * 100) if high_confidence else None

        # Down-capture
        low_confidence = [r for r in valid if r[prob_key] < 40]
        down_hit_rate = (sum(1 for r in low_confidence if (r[ret_key] or 0) <= 0) /
                         len(low_confidence) * 100) if low_confidence else None

        # Brier score
        brier = mean(
            ((r[prob_key] / 100) - (1 if (r[ret_key] or 0) > 0 else 0)) ** 2
            for r in valid
        )

        # Ranking test
        sorted_by_prob = sorted(valid, key=lambda r: r[prob_key], reverse=True)
        q_size = max(1, n // 4)
        top_q = sorted_by_prob[:q_size]
        bot_q = sorted_by_prob[-q_size:]
        top_avg_ret = mean(r[ret_key] for r in top_q if r[ret_key] is not None)
        bot_avg_ret = mean(r[ret_key] for r in bot_q if r[ret_key] is not None)
        top_q_up = sum(1 for r in top_q if (r[ret_key] or 0) > 0) / max(1, len([r for r in top_q if r[ret_key] is not None])) * 100

        avg_prob = mean(r[prob_key] for r in valid)
        avg_ret = mean(r[ret_key] for r in valid if r[ret_key] is not None)
        up_ratio = sum(1 for r in valid if (r[ret_key] or 0) > 0) / n * 100

        # Raw (uncalibrated) accuracy
        raw_probs = [r for r in valid if f"raw_{horizon}" in r]
        if raw_probs:
            raw_correct = sum(
                1 for r in raw_probs
                if (r[f"raw_{horizon}"] >= 50 and (r[ret_key] or 0) > 0) or
                   (r[f"raw_{horizon}"] < 50 and (r[ret_key] or 0) <= 0)
            )
            raw_acc = raw_correct / len(raw_probs) * 100
        else:
            raw_acc = None

        metrics[horizon] = {
            "n": n,
            "avg_prob": round(avg_prob, 1),
            "avg_ret": round(avg_ret, 2),
            "up_ratio": round(up_ratio, 1),
            "direction_acc": round(direction_acc, 1),
            "raw_direction_acc": round(raw_acc, 1) if raw_acc is not None else None,
            "up_hit_rate": round(up_hit_rate, 1) if up_hit_rate is not None else None,
            "down_hit_rate": round(down_hit_rate, 1) if down_hit_rate is not None else None,
            "brier_score": round(brier, 4),
            "top_q_avg_ret": round(top_avg_ret, 2),
            "bot_q_avg_ret": round(bot_avg_ret, 2),
            "top_q_up_pct": round(top_q_up, 1),
        }

    return {
        "test_date": test_date,
        "regime": regime.get("regime", "?"),
        "regime_score": regime.get("score", 50),
        "n_total": len(results),
        "failures": failures,
        "elapsed": round(elapsed, 1),
        "metrics": metrics,
    }


# ── Comparison mode: old vs new ─────────────────────────────────────────

def run_backtest_baseline(test_date: str, max_stocks: int = 150,
                          rate_limit: float = 5.0) -> Dict:
    """Run OLD baseline (no regime, no calibration, no divergence, no z-score)."""
    import baostock as bs

    t0 = time.monotonic()
    print(f"\n{'='*60}")
    print(f"Baseline Backtest: {test_date}")
    print(f"{'='*60}")

    df = fetch_universe()
    candidates = []
    for _, row in df.iterrows():
        code = str(row.iloc[0]).strip()
        name = str(row.iloc[1]).strip()
        if filter_basic(code, name):
            candidates.append((code, name))

    if max_stocks > 0:
        candidates = candidates[:max_stocks]

    results = []
    failures = 0

    bs.login()
    try:
        for i, (code, name) in enumerate(candidates):
            if (i + 1) % 20 == 0:
                elapsed = time.monotonic() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  {i+1}/{len(candidates)} ({rate:.1f}/s) | {len(results)} predicted",
                      file=sys.stderr, flush=True)

            time.sleep(1.0 / rate_limit)

            fin = fetch_one_stock(code, name)
            if fin is None:
                failures += 1
                continue

            end_dt = datetime.strptime(test_date, "%Y-%m-%d")
            daily = fetch_klines(bs, code, "d",
                                 start=(end_dt - timedelta(days=4*365)).strftime("%Y-%m-%d"),
                                 end=test_date)
            weekly = fetch_klines(bs, code, "w",
                                  start=(end_dt - timedelta(days=6*365)).strftime("%Y-%m-%d"),
                                  end=test_date)
            monthly = fetch_klines(bs, code, "m",
                                   start=(end_dt - timedelta(days=10*365)).strftime("%Y-%m-%d"),
                                   end=test_date)

            ind_d = compute_timeframe_indicators(daily)
            ind_w = compute_timeframe_indicators(weekly)
            ind_m = compute_timeframe_indicators(monthly)

            if ind_d is None:
                failures += 1
                continue

            tech_d = score_timeframe(ind_d)
            tech_w = score_timeframe(ind_w) if ind_w else 50
            tech_m = score_timeframe(ind_m) if ind_m else 50

            # Old-style fin_score (simple average, no z-score, no calibration)
            fin_score = (sum([
                _score_simple(fin, "profit"), _score_simple(fin, "growth"),
                _score_simple(fin, "health"), _score_simple(fin, "moat"),
            ]) / 4 * 10) if fin else 50

            mtf_aligned = 0
            if ind_d and ind_w:
                if ind_d["above_ma20"] and ind_w["above_ma20"]:
                    mtf_aligned += 1
                if ind_d["macd_above_zero"] and ind_w["macd_above_zero"]:
                    mtf_aligned += 1
            if ind_w and ind_m:
                if ind_w["above_ma20"] and ind_m["above_ma20"]:
                    mtf_aligned += 1

            resonance_bonus = mtf_aligned * 3

            prob_1m = min(95, tech_d * 0.70 + tech_w * 0.30 + resonance_bonus * 0.5)
            prob_2m = min(95, tech_d * 0.35 + tech_w * 0.35 + fin_score * 0.30 + resonance_bonus * 0.5)
            prob_3m = min(95, tech_w * 0.50 + fin_score * 0.50 + resonance_bonus * 0.3)
            prob_6m = min(95, tech_m * 0.40 + fin_score * 0.60 + resonance_bonus * 0.3)

            pred = {
                "code": code, "name": name,
                "prob_1m": round(prob_1m, 1), "prob_2m": round(prob_2m, 1),
                "prob_3m": round(prob_3m, 1), "prob_6m": round(prob_6m, 1),
                "fin_score": round(fin_score, 1),
            }

            actual = {}
            for label, days in [("1m", 21), ("2m", 42), ("3m", 63), ("6m", 126)]:
                ret = measure_actual_return(bs, code, test_date, days)
                actual[f"ret_{label}"] = ret

            if actual["ret_1m"] is None:
                continue

            results.append({**pred, **actual})
    finally:
        bs.logout()

    elapsed = time.monotonic() - t0

    metrics = {}
    for horizon in ["1m", "2m", "3m", "6m"]:
        prob_key = f"prob_{horizon}"
        ret_key = f"ret_{horizon}"
        valid = [r for r in results if r.get(ret_key) is not None]
        if len(valid) < 5:
            metrics[horizon] = {"error": f"only {len(valid)} samples"}
            continue
        n = len(valid)
        correct = sum(1 for r in valid if (r[prob_key] >= 50 and (r[ret_key] or 0) > 0) or (r[prob_key] < 50 and (r[ret_key] or 0) <= 0))
        sorted_by_prob = sorted(valid, key=lambda r: r[prob_key], reverse=True)
        q_size = max(1, n // 4)
        top_q = sorted_by_prob[:q_size]
        bot_q = sorted_by_prob[-q_size:]
        metrics[horizon] = {
            "n": n,
            "direction_acc": round(correct / n * 100, 1),
            "top_q_avg_ret": round(mean(r[ret_key] for r in top_q if r[ret_key] is not None), 2),
            "bot_q_avg_ret": round(mean(r[ret_key] for r in bot_q if r[ret_key] is not None), 2),
        }

    return {"test_date": test_date, "n_total": len(results), "metrics": metrics}


def _score_simple(m, dim_name):
    """Simple absolute score without industry z-score, for baseline comparison."""
    if dim_name == "profit":
        roe = m.get("roe_latest", 0) or 0
        if roe >= 25: return 9
        elif roe >= 20: return 8
        elif roe >= 15: return 7
        elif roe >= 10: return 6
        elif roe >= 5: return 4
        else: return 2
    elif dim_name == "growth":
        r = m.get("revenue_cagr_3y", 0) or 0
        if r >= 25: return 8
        elif r >= 15: return 7
        elif r >= 10: return 6
        elif r >= 5: return 5
        elif r >= 0: return 4
        else: return 2
    elif dim_name == "health":
        d = m.get("debt_ratio", 50) or 50
        if d < 20: return 9
        elif d < 30: return 8
        elif d < 40: return 7
        elif d < 50: return 6
        elif d < 60: return 5
        else: return 3
    elif dim_name == "moat":
        gm = m.get("gross_margin", 0) or 0
        if gm >= 60: return 9
        elif gm >= 40: return 7
        elif gm >= 20: return 5
        else: return 3
    return 5


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Backtest enhanced trend prediction model")
    p.add_argument("--date", type=str, default=None, help="Test date YYYY-MM-DD")
    p.add_argument("--max", type=int, default=150, help="Max stocks to test")
    p.add_argument("--rate", type=float, default=5.0, help="Queries/sec")
    p.add_argument("--baseline", action="store_true", help="Run old baseline for comparison")
    p.add_argument("--compare", action="store_true", help="Run both enhanced and baseline")
    args = p.parse_args()

    today = datetime.now()
    if args.date:
        test_dates = [args.date]
    else:
        test_dates = [
            (today - timedelta(days=90)).strftime("%Y-%m-%d"),
            (today - timedelta(days=180)).strftime("%Y-%m-%d"),
        ]

    if args.compare:
        print("\n" + "="*100)
        print("RUNNING COMPARISON: ENHANCED vs BASELINE")
        print("="*100)

        for td in test_dates:
            enhanced = run_backtest(td, max_stocks=args.max, rate_limit=args.rate)
            baseline = run_backtest_baseline(td, max_stocks=args.max, rate_limit=args.rate)

            print(f"\n{'='*100}")
            print(f"COMPARISON: {td} ({enhanced.get('regime', '?')} market)")
            print(f"{'='*100}")
            print(f"{'Horizon':<8} {'类型':<6} {'N':>4} {'DirAcc':>8} {'TopQ':>8} {'BotQ':>8} {'利差':>8}")
            print("-" * 60)
            for h in ["1m", "2m", "3m", "6m"]:
                em = enhanced["metrics"].get(h, {})
                bm = baseline["metrics"].get(h, {})
                if "error" not in em and "error" not in bm:
                    e_spread = em["top_q_avg_ret"] - em["bot_q_avg_ret"]
                    b_spread = bm["top_q_avg_ret"] - bm["bot_q_avg_ret"]
                    print(f"  {h:<6} 增强版 {em['n']:>4} {em['direction_acc']:>7.1f}% {em['top_q_avg_ret']:>7.2f}% {em['bot_q_avg_ret']:>7.2f}% {e_spread:>7.2f}%")
                    print(f"  {h:<6} 基线版 {bm['n']:>4} {bm['direction_acc']:>7.1f}% {bm['top_q_avg_ret']:>7.2f}% {bm['bot_q_avg_ret']:>7.2f}% {b_spread:>7.2f}%")
                    da_delta = em['direction_acc'] - bm['direction_acc']
                    sp_delta = e_spread - b_spread
                    print(f"         {'':>6} {'Δ':>4} {da_delta:>+7.1f}% {'':>8} {'':>8} {sp_delta:>+7.2f}%")
    else:
        all_results = []
        for td in test_dates:
            bt = run_backtest(td, max_stocks=args.max, rate_limit=args.rate)
            all_results.append(bt)

        print(f"\n{'='*100}")
        print("ENHANCED BACKTEST SUMMARY")
        print(f"{'='*100}")

        for bt in all_results:
            print(f"\n── {bt['test_date']} | Regime: {bt.get('regime', '?')} (score={bt.get('regime_score', '?')}) | {bt['n_total']} stocks, {bt['elapsed']}s ──")
            print(f"{'Horizon':<8} {'N':>4} {'AvgProb':>7} {'AvgRet':>7} "
                  f"{'Up%':>6} {'DirAcc':>7} {'RawAcc':>7} {'UpHit':>7} {'DownHit':>8} "
                  f"{'Brier':>6} {'TopQ':>7} {'BotQ':>7} {'利差':>7}")
            print("-" * 105)
            for h in ["1m", "2m", "3m", "6m"]:
                m = bt["metrics"].get(h, {})
                if "error" in m:
                    print(f"  {h:<6}  {m['error']}")
                    continue
                spread = m['top_q_avg_ret'] - m['bot_q_avg_ret']
                uh = m.get('up_hit_rate') or '-'
                dh = m.get('down_hit_rate') or '-'
                ra = m.get('raw_direction_acc')
                print(f"  {h:<6}  {m['n']:>4}  {m['avg_prob']:>5.1f}% {m['avg_ret']:>6.2f}% "
                      f"{m['up_ratio']:>5.1f}% {m['direction_acc']:>6.1f}% "
                      f"{ra or '-':>6}% "
                      f"{uh:>6}% {dh:>7}% "
                      f"{m['brier_score']:>6.4f} {m['top_q_avg_ret']:>6.2f}% {m['bot_q_avg_ret']:>6.2f}% {spread:>+6.2f}%")

        # Aggregate
        print(f"\n── AGGREGATE ACROSS {len(all_results)} TEST DATES ──")
        for h in ["1m", "2m", "3m", "6m"]:
            valid = [bt["metrics"][h] for bt in all_results if h in bt["metrics"] and "error" not in bt["metrics"][h]]
            if not valid:
                continue
            avg_dir = mean(m["direction_acc"] for m in valid)
            avg_brier = mean(m["brier_score"] for m in valid)
            spreads = [m["top_q_avg_ret"] - m["bot_q_avg_ret"] for m in valid]
            avg_spread = mean(spreads)
            print(f"  {h}: DirAcc={avg_dir:.1f}% Brier={avg_brier:.4f} TopQ-BotQ利差={avg_spread:+.2f}% 样本={sum(m['n'] for m in valid)}")
