#!/usr/bin/env python3
"""Multi-timeframe trend prediction engine.

Predicts upside probability for 1M / 2M / 3M / 6M+ horizons
by combining technical indicators (daily/weekly/monthly K-lines)
with fundamental quality scores.

Reuses screen_stocks.py for financial data and filtering.
"""

import sys
import os
import time
from statistics import mean
from typing import Optional, Dict, Any, List, Tuple

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, os.path.join(SCRIPTS_DIR, "lib"))

from screen_stocks import (
    fetch_universe, filter_basic, filter_financial,
    fetch_one_stock, score_stock,
)
from policy_alpha import fetch_industry, fetch_growth_data, get_policy_score
from market_regime import get_regime

# ── K-line fetching ──────────────────────────────────────────────────────

def fetch_klines(bs, code: str, frequency: str = "d",
                 start: str = "2021-01-01", end: str = "") -> dict:
    """Fetch OHLCV data via baostock.

    Returns dict with lists: dates, opens, highs, lows, closes, volumes.
    frequency: 'd' (daily), 'w' (weekly), 'm' (monthly).
    """
    fields = "date,open,high,low,close,volume"
    rs = bs.query_history_k_data_plus(
        code, fields,
        start_date=start,
        end_date=end or time.strftime("%Y-%m-%d"),
        frequency=frequency,
        adjustflag="2",  # forward-adjusted
    )
    if rs.error_code != "0":
        return {"dates": [], "opens": [], "highs": [], "lows": [], "closes": [], "volumes": []}

    data = {"dates": [], "opens": [], "highs": [], "lows": [], "closes": [], "volumes": []}
    while rs.next():
        row = rs.get_row_data()
        try:
            data["dates"].append(row[0])
            data["opens"].append(float(row[1]) if row[1] else 0)
            data["highs"].append(float(row[2]) if row[2] else 0)
            data["lows"].append(float(row[3]) if row[3] else 0)
            data["closes"].append(float(row[4]) if row[4] else 0)
            data["volumes"].append(float(row[5]) if row[5] else 0)
        except (ValueError, IndexError):
            continue
    return data


# ── Technical indicator helpers ──────────────────────────────────────────

def _ma(values, n):
    """Simple moving average."""
    if len(values) < n:
        return []
    out = []
    for i in range(len(values)):
        window = values[max(0, i - n + 1):i + 1]
        out.append(sum(window) / len(window))
    return out


def _ema(values, n):
    """Exponential moving average."""
    if not values:
        return []
    k = 2 / (n + 1)
    out = []
    prev = None
    for v in values:
        prev = v if prev is None else v * k + prev * (1 - k)
        out.append(prev)
    return out


def _rsi(closes, n=14):
    """Relative Strength Index."""
    if len(closes) < n + 1:
        return 50
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = mean(gains[-n:])
    avg_loss = mean(losses[-n:]) or 1e-9
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def _adx(highs, lows, closes, n=14):
    """Average Directional Index — trend strength indicator.

    ADX > 25: strong trend (trust trend signals)
    ADX 20-25: moderate trend
    ADX < 20: weak/choppy (trust mean-reversion signals)
    """
    if len(closes) < n * 2:
        return 20  # neutral default
    tr_list, plus_dm_list, minus_dm_list = [], [], []
    for i in range(1, len(closes)):
        h, l, pc = highs[i], lows[i], closes[i-1]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        tr_list.append(tr)
        up = h - highs[i-1]
        dn = lows[i-1] - l
        plus_dm_list.append(up if up > 0 and up > dn else 0)
        minus_dm_list.append(dn if dn > 0 and dn > up else 0)
    atr = _rma(tr_list, n)
    plus_di = [100 * (p / a) if a > 0 else 0 for p, a in zip(_rma(plus_dm_list, n), atr)]
    minus_di = [100 * (m / a) if a > 0 else 0 for m, a in zip(_rma(minus_dm_list, n), atr)]
    dx = [100 * abs(p - m) / (p + m) if (p + m) > 0 else 0 for p, m in zip(plus_di, minus_di)]
    adx = _rma(dx, n)
    return adx[-1] if adx else 20


def _rma(values, n):
    """Wilder's smoothed moving average (RMA)."""
    if not values or len(values) < n:
        return values[:] if values else []
    out = [sum(values[:n]) / n]
    for v in values[n:]:
        out.append((out[-1] * (n - 1) + v) / n)
    return out


def _stage(closes, ma200):
    """Weinstein Stage Analysis: 1=底部 2=上升 3=顶部 4=下跌."""
    if len(closes) < 60 or not ma200 or len(ma200) < 60:
        return 0
    last = closes[-1]
    ma_now = ma200[-1]
    ma_60ago = ma200[-60]
    above = last > ma_now
    rising = ma_now > ma_60ago
    if above and rising:
        return 2
    if not above and rising:
        return 1
    if above and not rising:
        return 3
    return 4


def _detect_divergences(closes, highs, lows, dif_series, rsi_series=None):
    """Detect MACD and RSI divergences over recent 20-40 bars.

    Bearish divergence: price makes higher high but indicator makes lower high.
    Bullish divergence: price makes lower low but indicator makes higher low.

    Returns (macd_divergence, rsi_divergence) where:
      1 = bullish divergence, -1 = bearish divergence, 0 = none.
    """
    n = len(closes)
    if n < 30:
        return 0, 0

    macd_div = 0
    rsi_div = 0

    # Look at two segments: recent (last 5-15 bars) vs prior (15-35 bars ago)
    recent = slice(-15, None)
    prior = slice(-35, -15)

    # ── MACD divergence ──
    if len(dif_series) >= 35:
        # Bearish: price higher high, MACD lower high
        price_recent_high = max(highs[recent])
        price_prior_high = max(highs[prior])
        macd_recent_high = max(dif_series[recent])
        macd_prior_high = max(dif_series[prior])

        if price_recent_high > price_prior_high and macd_recent_high < macd_prior_high:
            macd_div = -1  # bearish divergence

        # Bullish: price lower low, MACD higher low
        price_recent_low = min(lows[recent])
        price_prior_low = min(lows[prior])
        macd_recent_low = min(dif_series[recent])
        macd_prior_low = min(dif_series[prior])

        if price_recent_low < price_prior_low and macd_recent_low > macd_prior_low:
            macd_div = 1  # bullish divergence

    # ── RSI divergence ──
    if rsi_series is None:
        # Compute RSI values for key points
        rsi_recent = _rsi(closes[recent], 14) if len(closes[recent]) > 14 else 50
        rsi_prior = _rsi(closes[:len(closes)-15], 14) if len(closes) > 29 else 50
    else:
        rsi_recent = rsi_series
        rsi_prior = rsi_series  # approximation: use same RSI

    # Better approach: compute RSI at the prior high/low point
    # Simplified: check if recent RSI diverges from recent price action
    if len(closes) >= 20:
        # Bearish RSI divergence: price near highs but RSI < 70 and falling
        n_lookback = min(20, len(closes))
        recent_high = max(highs[-n_lookback:])
        near_high = closes[-1] >= recent_high * 0.97
        rsi_val = _rsi(closes, 14)
        rsi_20ago = _rsi(closes[:-10], 14) if len(closes) > 24 else rsi_val
        if near_high and rsi_val < 70 and rsi_val < rsi_20ago:
            rsi_div = -1  # bearish RSI divergence

        # Bullish RSI divergence: price near lows but RSI > 30 and rising
        recent_low = min(lows[-n_lookback:])
        near_low = closes[-1] <= recent_low * 1.03
        if near_low and rsi_val > 30 and rsi_val > rsi_20ago:
            rsi_div = 1  # bullish RSI divergence

    return macd_div, rsi_div


# ── Per-timeframe indicator computation ──────────────────────────────────

def compute_timeframe_indicators(klines: dict) -> dict:
    """Compute 6 technical signals for a single timeframe."""
    closes = klines["closes"]
    highs = klines["highs"]
    lows = klines["lows"]
    vols = klines["volumes"]

    if len(closes) < 60:
        return None

    last = closes[-1]

    # MAs
    ma5s = _ma(closes, 5)
    ma10s = _ma(closes, 10)
    ma20s = _ma(closes, 20)
    ma60s = _ma(closes, 60)
    ma120s = _ma(closes, 120)
    ma200s = _ma(closes, 200) if len(closes) >= 200 else []

    # MACD
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    dif = [a - b for a, b in zip(ema12, ema26)]
    dea = _ema(dif, 9)
    macd_hist = [(d - e) * 2 for d, e in zip(dif, dea)]

    # RSI
    rsi_val = _rsi(closes, 14)

    # Divergence detection (MACD + RSI)
    div_macd, div_rsi = _detect_divergences(closes, highs, lows, dif, rsi_val)

    # Stage
    stg = _stage(closes, ma200s)

    # Volume: last 5 vs last 20 average
    avg_vol_5 = mean(vols[-5:]) if len(vols) >= 5 else 0
    avg_vol_20 = mean(vols[-20:]) if len(vols) >= 20 else 0
    vol_ratio = avg_vol_5 / avg_vol_20 if avg_vol_20 > 0 else 1

    # Price position vs 1-year range (or 250-bar range)
    n_lookback = min(250, len(closes))
    yr_high = max(highs[-n_lookback:])
    yr_low = min(lows[-n_lookback:])
    pct_from_high = (last - yr_high) / yr_high * 100 if yr_high > 0 else 0
    pct_range = (last - yr_low) / (yr_high - yr_low) * 100 if yr_high > yr_low else 50

    return {
        "last_close": last,
        "ma5": ma5s[-1] if ma5s else 0,
        "ma10": ma10s[-1] if ma10s else 0,
        "ma20": ma20s[-1] if ma20s else 0,
        "ma60": ma60s[-1] if ma60s else 0,
        "ma120": ma120s[-1] if ma120s else 0,
        "ma200": ma200s[-1] if ma200s else 0,
        # Bull alignment: MA5 > MA10 > MA20 > MA60 > MA120
        "ma_bull_aligned": (
            ma5s[-1] > ma10s[-1] > ma20s[-1] > ma60s[-1] > ma120s[-1]
            if all([ma5s, ma10s, ma20s, ma60s, ma120s]) else False
        ),
        # Partial alignment (price above key MAs)
        "above_ma20": last > ma20s[-1] if ma20s else False,
        "above_ma60": last > ma60s[-1] if ma60s else False,
        "above_ma200": last > ma200s[-1] if ma200s else False,
        "ma20_slope": (ma20s[-1] - ma20s[-20]) / ma20s[-20] * 100 if len(ma20s) >= 20 and ma20s[-20] > 0 else 0,
        # MACD
        "macd_dif": dif[-1] if dif else 0,
        "macd_dea": dea[-1] if dea else 0,
        "macd_hist": macd_hist[-1] if macd_hist else 0,
        "macd_golden": dif[-1] > dea[-1] and dif[-2] <= dea[-2] if len(dif) >= 2 and len(dea) >= 2 else False,
        "macd_above_zero": dif[-1] > 0 if dif else False,
        "macd_divergence": div_macd,  # -1 bearish, 0 none, +1 bullish
        # RSI
        "rsi": rsi_val,
        "rsi_divergence": div_rsi,  # -1 bearish, 0 none, +1 bullish
        # Stage
        "stage": stg,
        # Volume
        "vol_ratio": vol_ratio,
        "price_up_vol_up": last > closes[-2] and vol_ratio > 1.2 if len(closes) >= 2 else False,
        # Price position
        "pct_from_high": pct_from_high,
        "pct_range": pct_range,
        # Recent momentum
        "ret_1m": (closes[-1] / closes[-min(20, len(closes))] - 1) * 100 if len(closes) >= 20 else 0,
        "ret_3m": (closes[-1] / closes[-min(60, len(closes))] - 1) * 100 if len(closes) >= 60 else 0,
        # ADX trend strength
        "adx": round(_adx(highs, lows, closes), 1),
        # Bar count
        "n_bars": len(closes),
    }


# ── Timeframe scoring ────────────────────────────────────────────────────

def score_timeframe(ind: dict) -> float:
    """Score a single timeframe 0-10, return 0-100."""
    if ind is None:
        return 50  # neutral for insufficient data

    s = 0

    # 1. MA alignment (0-2)
    if ind["ma_bull_aligned"]:
        s += 2
    elif ind["above_ma20"] and ind["above_ma60"]:
        s += 1.5
    elif ind["above_ma20"]:
        s += 1
    elif ind["above_ma200"]:
        s += 0.5

    # 2. MACD (0-2)
    if ind["macd_above_zero"] and ind["macd_golden"]:
        s += 2
    elif ind["macd_above_zero"] and ind["macd_dif"] > ind["macd_dea"]:
        s += 1.5
    elif ind["macd_above_zero"]:
        s += 1
    elif ind["macd_golden"]:
        s += 0.5

    # Divergence adjustment (-1 to +1)
    div_macd = ind.get("macd_divergence", 0)
    div_rsi = ind.get("rsi_divergence", 0)
    if div_macd == -1:
        s -= 0.5  # bearish MACD divergence
    elif div_macd == 1:
        s += 0.5  # bullish MACD divergence
    if div_rsi == -1:
        s -= 0.3  # bearish RSI divergence
    elif div_rsi == 1:
        s += 0.3  # bullish RSI divergence

    # 3. RSI (0-2)
    rsi = ind["rsi"]
    if 40 <= rsi <= 70:
        s += 2  # healthy trending
    elif 30 <= rsi < 40:
        s += 1.5  # pullback, may bounce
    elif 70 < rsi <= 80:
        s += 1  # extended but still strong
    elif rsi < 30:
        s += 1  # oversold, potential reversal
    else:
        s += 0.5  # >80 overbought

    # 4. Stage (0-2)
    stage = ind["stage"]
    if stage == 2:
        s += 2
    elif stage == 1:
        s += 1.5
    elif stage == 3:
        s += 0.5
    elif stage == 4:
        s += 0
    else:
        s += 1  # unknown

    # 5. Volume/price relationship (0-1)
    if ind["price_up_vol_up"]:
        s += 1
    elif ind["vol_ratio"] > 0.8:
        s += 0.5

    # 6. Price position (0-1)
    pct = abs(ind["pct_from_high"])
    if pct < 5:
        s += 1  # near highs, strong momentum
    elif pct < 15:
        s += 0.7
    elif pct < 25:
        s += 0.4
    else:
        s += 0.1  # far from highs, weak

    # 7. Absolute momentum (0-1)
    ret_1m = ind.get("ret_1m", 0) or 0
    if ret_1m > 15:
        s += 1.0
    elif ret_1m > 8:
        s += 0.7
    elif ret_1m > 3:
        s += 0.5
    elif ret_1m < -15:
        s -= 1.0
    elif ret_1m < -8:
        s -= 0.7
    elif ret_1m < -3:
        s -= 0.5

    # 8. ADX trend strength adjustment (-1.5 to +1.0)
    adx = ind.get("adx", 20) or 20
    if adx > 30:
        s += 1.0   # strong trend — trend signals are reliable
    elif adx > 25:
        s += 0.5   # moderate trend
    elif adx < 15:
        s -= 1.5   # very choppy — trend signals are noise
    elif adx < 20:
        s -= 1.0   # choppy — reduce confidence

    # Convert 0-10 scale to 0-100
    return (s / 10) * 100


# ── Stock-level prediction ───────────────────────────────────────────────

def calibrate_prob(raw_prob: float, horizon: str, regime_score: int) -> float:
    """Calibrate raw probability using backtest-derived correction.

    Based on backtest results:
    - Strong market (regime_score >= 65): minimal calibration
    - Sideways (40-64): moderate calibration
    - Weak market (< 40): aggressive downward calibration

    The calibration is stronger for shorter horizons (more noise).
    """
    # Determine market state
    if regime_score >= 65:
        # Strong: model is roughly calibrated, slight adjustments
        if horizon == "1m":
            calibrated = raw_prob * 0.95 + 2.5
        elif horizon == "2m":
            calibrated = raw_prob * 0.95 + 2.5
        elif horizon == "3m":
            calibrated = raw_prob * 1.0 + 0.0
        else:  # 6m
            calibrated = raw_prob * 1.0 + 0.0
    elif regime_score >= 40:
        # Sideways: moderate dampening
        if horizon == "1m":
            calibrated = raw_prob * 0.80 + 10.0
        elif horizon == "2m":
            calibrated = raw_prob * 0.82 + 9.0
        elif horizon == "3m":
            calibrated = raw_prob * 0.85 + 7.5
        else:  # 6m
            calibrated = raw_prob * 0.88 + 6.0
    else:
        # Weak/bear: aggressive dampening based on backtest
        # Feb 2026 data: raw ~60-63% → actual ~20-31%
        bear_depth = 40 - regime_score  # 0 to 40
        dampen = 0.65 - bear_depth * 0.005  # 0.65 down to 0.45

        if horizon == "1m":
            calibrated = raw_prob * dampen + 5.0
        elif horizon == "2m":
            calibrated = raw_prob * dampen + 6.0
        elif horizon == "3m":
            calibrated = raw_prob * (dampen + 0.05) + 5.0
        else:  # 6m
            calibrated = raw_prob * (dampen + 0.08) + 5.0

    return min(95, max(5, calibrated))


def predict_one_stock(bs, code: str, name: str,
                      fin_data: Optional[Dict] = None,
                      regime: Optional[Dict] = None) -> Optional[Dict]:
    """Run full multi-timeframe prediction for one stock.

    Returns dict with 1M/2M/3M/6M+ probabilities and diagnostic data,
    or None if insufficient data.
    """
    # Fetch daily/weekly/monthly K-lines
    daily = fetch_klines(bs, code, "d", start="2022-01-01")
    weekly = fetch_klines(bs, code, "w", start="2020-01-01")
    monthly = fetch_klines(bs, code, "m", start="2018-01-01")

    ind_d = compute_timeframe_indicators(daily)
    ind_w = compute_timeframe_indicators(weekly)
    ind_m = compute_timeframe_indicators(monthly)

    # Need at least daily indicators
    if ind_d is None:
        return None

    tech_d = score_timeframe(ind_d)
    tech_w = score_timeframe(ind_w) if ind_w else 50
    tech_m = score_timeframe(ind_m) if ind_m else 50

    # Financial score (0-100): average of 9 dimensions
    if fin_data:
        scored = score_stock(fin_data)
        dim_scores = [
            scored["profitability"], scored["growth"], scored["health"],
            scored["valuation"], scored["moat"], scored["policy"],
            scored["institutional"], scored["cycle"], scored["controller"],
        ]
        fin_score = sum(dim_scores) / 9 * 10
    else:
        fin_score = 50
        scored = {}

    # Multi-timeframe resonance bonus
    mtf_aligned = 0
    if ind_d and ind_w:
        if ind_d["above_ma20"] and ind_w["above_ma20"]:
            mtf_aligned += 1
        if ind_d["macd_above_zero"] and ind_w["macd_above_zero"]:
            mtf_aligned += 1
    if ind_w and ind_m:
        if ind_w["above_ma20"] and ind_m["above_ma20"]:
            mtf_aligned += 1

    resonance_bonus = mtf_aligned * 3  # up to +9 bonus

    # Horizon predictions
    # 1M: daily 70% + weekly 30%
    prob_1m = tech_d * 0.70 + tech_w * 0.30
    prob_1m = min(95, prob_1m + resonance_bonus * 0.5)

    # 2M: daily 35% + weekly 35% + fundamental 30%
    prob_2m = tech_d * 0.35 + tech_w * 0.35 + fin_score * 0.30
    prob_2m = min(95, prob_2m + resonance_bonus * 0.5)

    # 3M: weekly 50% + fundamental 50%
    prob_3m = tech_w * 0.50 + fin_score * 0.50
    prob_3m = min(95, prob_3m + resonance_bonus * 0.3)

    # 6M+: monthly 40% + fundamental 60%
    prob_6m = tech_m * 0.40 + fin_score * 0.60
    prob_6m = min(95, prob_6m + resonance_bonus * 0.3)

    # ── Horizon decay blending ──
    # Short-horizon predictions are noisy; anchor them with longer horizons.
    # Backtest shows 3M/6M are most reliable (59% direction accuracy at 3M).
    raw_1m = prob_1m
    raw_2m = prob_2m
    prob_1m = raw_1m * 0.25 + prob_3m * 0.40 + prob_6m * 0.35
    prob_2m = raw_2m * 0.30 + prob_3m * 0.35 + prob_6m * 0.35
    # 3M and 6M unchanged — they are the anchors

    # ADX for diagnostics
    adx_d = ind_d.get("adx", 20) or 20 if ind_d else 20

    # ── Market regime (display only, no probability modification) ──
    if regime is None:
        regime = get_regime(bs)
    regime_score = regime.get("score", 50)

    # ── Relative Strength (RS) alpha ──
    # Compare stock returns vs index returns to identify genuine outperformance.
    index_ret_1m = regime.get("ret_1m", 0) or 0
    index_ret_3m = regime.get("ret_3m", 0) or 0
    stock_ret_1m = ind_d.get("ret_1m", 0) or 0 if ind_d else 0
    stock_ret_3m = ind_d.get("ret_3m", 0) or 0 if ind_d else 0
    alpha_1m = stock_ret_1m - index_ret_1m  # stock outperformance vs index
    alpha_3m = stock_ret_3m - index_ret_3m

    # RS alpha bonus: stocks beating the index get a lift
    if alpha_1m > 10:
        rs_bonus_1m = 2.0
    elif alpha_1m > 5:
        rs_bonus_1m = 1.0
    elif alpha_1m < -10:
        rs_bonus_1m = -2.0
    elif alpha_1m < -5:
        rs_bonus_1m = -1.0
    else:
        rs_bonus_1m = 0

    if alpha_3m > 15:
        rs_bonus_3m = 1.5
    elif alpha_3m > 8:
        rs_bonus_3m = 0.8
    elif alpha_3m < -15:
        rs_bonus_3m = -1.5
    elif alpha_3m < -8:
        rs_bonus_3m = -0.8
    else:
        rs_bonus_3m = 0

    prob_1m = min(95, max(5, prob_1m + rs_bonus_1m))
    prob_2m = min(95, max(5, prob_2m + rs_bonus_1m * 0.7 + rs_bonus_3m * 0.3))
    prob_3m = min(95, max(5, prob_3m + rs_bonus_3m))
    prob_6m = min(95, max(5, prob_6m + rs_bonus_3m * 0.5))

    return {
        "code": fin_data.get("code", code) if fin_data else code,
        "name": fin_data.get("name", name) if fin_data else name,
        "prob_1m": round(prob_1m, 1),
        "prob_2m": round(prob_2m, 1),
        "prob_3m": round(prob_3m, 1),
        "prob_6m": round(prob_6m, 1),
        # Diagnostics
        "tech_d": round(tech_d, 1),
        "tech_w": round(tech_w, 1),
        "tech_m": round(tech_m, 1),
        "fin_score": round(fin_score, 1),
        # 9-dimension fundamental scores (0-10)
        "profitability": round(scored.get("profitability", 5), 1),
        "growth": round(scored.get("growth", 5), 1),
        "health": round(scored.get("health", 5), 1),
        "valuation": round(scored.get("valuation", 5), 1),
        "moat": round(scored.get("moat", 5), 1),
        "policy": round(scored.get("policy", 5), 1),
        "institutional": round(scored.get("institutional", 5), 1),
        "cycle": round(scored.get("cycle", 5), 1),
        "controller": round(scored.get("controller", 5), 1),
        "total": round(scored.get("total", 50), 1),
        "roe": fin_data.get("roe_latest", 0) if fin_data else 0,
        "debt": fin_data.get("debt_ratio", 50) if fin_data else 50,
        "stage_d": ind_d["stage"] if ind_d else 0,
        "stage_w": ind_w["stage"] if ind_w else 0,
        "ma_bull_d": ind_d["ma_bull_aligned"] if ind_d else False,
        "macd_golden_d": ind_d["macd_golden"] if ind_d else False,
        "rsi_d": ind_d["rsi"] if ind_d else 50,
        "vol_up_d": ind_d["price_up_vol_up"] if ind_d else False,
        "mtf_resonance": mtf_aligned,
        "policy_name": fin_data.get("policy_name", "") if fin_data else "",
        "policy_score": fin_data.get("policy_score", 0) if fin_data else 0,
        "inst_summary": fin_data.get("inst_summary", "") if fin_data else "",
        "cycle_stage": fin_data.get("cycle_stage", "—") if fin_data else "—",
        "seasonal_phase": fin_data.get("seasonal_phase", "—") if fin_data else "—",
        "ctrl_type": fin_data.get("ctrl_type", "—") if fin_data else "—",
        "ctrl_north_south": fin_data.get("ctrl_north_south", "—") if fin_data else "—",
        # Market regime
        "regime": regime.get("regime", "未知"),
        "regime_score": regime_score,
        # Divergence signals
        "div_macd_d": ind_d.get("macd_divergence", 0) if ind_d else 0,
        "div_rsi_d": ind_d.get("rsi_divergence", 0) if ind_d else 0,
        "div_macd_w": ind_w.get("macd_divergence", 0) if ind_w else 0,
        # Momentum (for industry-relative ranking)
        "ret_1m": round(ind_d.get("ret_1m", 0) or 0, 1) if ind_d else 0,
        "ret_3m": round(ind_d.get("ret_3m", 0) or 0, 1) if ind_d else 0,
        "industry": fin_data.get("industry", "") if fin_data else "",
        # RS alpha
        "alpha_1m": round(alpha_1m, 1),
        "alpha_3m": round(alpha_3m, 1),
        # ADX
        "adx_d": round(adx_d, 1),
    }


# ── Batch prediction pipeline ────────────────────────────────────────────

def predict_batch(max_stocks: int = 0, top_n: int = 30, rate_limit: float = 5.0,
                  horizon_sort: str = "prob_6m") -> Tuple[List[Dict], Dict]:
    """Run batch trend prediction. Returns (rankings, stats)."""
    import baostock as bs

    t0 = time.monotonic()
    print("Fetching A-share universe...", file=sys.stderr, flush=True)
    df = fetch_universe()

    # Tier 1 filter
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
    no_data = 0

    # Process within single baostock session
    bs.login()
    try:
        for i, (code, name) in enumerate(candidates):
            if (i + 1) % 25 == 0:
                elapsed = time.monotonic() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  {i+1}/{len(candidates)} ({rate:.1f}/s) | {len(results)} predicted",
                      file=sys.stderr, flush=True)

            time.sleep(1.0 / rate_limit)

            # Get financial data first
            fin = fetch_one_stock(code, name)
            if fin is None:
                failures += 1
                continue

            # Skip stocks that fail financial filter
            fail = filter_financial(fin)
            if fail:
                continue

            # Run multi-timeframe prediction
            pred = predict_one_stock(bs, code, name, fin)
            if pred is None:
                no_data += 1
                continue

            results.append(pred)
    finally:
        bs.logout()

    # Sort by chosen horizon
    results.sort(key=lambda r: r[horizon_sort], reverse=True)

    # ── Industry-relative momentum adjustment ──
    # Boost/penalize short-horizon probs based on within-industry momentum rank.
    # Stocks leading their industry get a lift; laggards get dampened.
    from industry_benchmark import compute_relative_momentum
    mom_map = compute_relative_momentum(results)
    for r in results:
        mom = mom_map.get(r["code"], {})
        adj = mom.get("combined", 0)
        # Apply to 1M/2M (most momentum-sensitive horizons)
        r["prob_1m"] = round(min(95, max(5, r["prob_1m"] + adj * 1.0)), 1)
        r["prob_2m"] = round(min(95, max(5, r["prob_2m"] + adj * 0.7)), 1)
        r["prob_3m"] = round(min(95, max(5, r["prob_3m"] + adj * 0.3)), 1)
        # Store momentum diagnostic
        r["mom_1m_pct"] = mom.get("ret_1m_pct", 50)
        r["mom_3m_pct"] = mom.get("ret_3m_pct", 50)
        r["mom_adj"] = adj
        r["mom_peers"] = mom.get("peer_count", 0)

    # Re-sort after momentum adjustment
    results.sort(key=lambda r: r[horizon_sort], reverse=True)
    results = results[:top_n]

    elapsed = time.monotonic() - t0
    stats.update({
        "elapsed": round(elapsed, 1),
        "predicted": len(results),
        "failures": failures,
        "no_data": no_data,
    })

    # Build recommended portfolio from full results (before truncation)
    portfolio = select_portfolio(results)
    stats["portfolio_n"] = len(portfolio)

    return results, stats, portfolio


# ── Portfolio selection ──────────────────────────────────────────────────

def select_portfolio(results: List[Dict], top_n: int = 12) -> List[Dict]:
    """Select a recommended buy portfolio from batch prediction results.

    Multi-signal resonance filter:
    1. prob_6m >= 55 (long-term bullish)
    2. alpha_1m > 0 (beating the index short-term)
    3. adx_d > 20 (trend exists)
    4. regime_score >= 40 (not bear market)
    5. total >= 55 (fundamentals pass)

    Falls back gradually if too few stocks pass all filters.
    """
    if not results:
        return []

    def _apply_filters(stocks, prob_min, alpha_min, adx_min, total_min):
        return [
            s for s in stocks
            if s["prob_6m"] >= prob_min
            and s.get("alpha_1m", 0) >= alpha_min
            and s.get("adx_d", 20) >= adx_min
            and s.get("total", 50) >= total_min
        ]

    # Tier 1: strict — all signals aligned
    portfolio = _apply_filters(results, 55, 0, 20, 55)

    # Tier 2: relax alpha (stock may be slightly lagging index)
    if len(portfolio) < 8:
        portfolio = _apply_filters(results, 52, -5, 20, 52)

    # Tier 3: relax ADX (weak trend OK if fundamentals strong)
    if len(portfolio) < 8:
        portfolio = _apply_filters(results, 50, -10, 15, 50)

    # Tier 4: bare minimum — just prob_6m and fundamentals
    if len(portfolio) < 5:
        portfolio = [s for s in results if s["prob_6m"] >= 48 and s.get("total", 0) >= 48]

    # Score and rank the filtered stocks
    # Composite: long-term prob (40%) + alpha strength (25%) + fundamentals (20%) + trend (15%)
    for s in portfolio:
        alpha_score = min(10, max(0, (s.get("alpha_1m", 0) + 15) / 3))  # map -15..+15 → 0..10
        adx_score = min(10, max(0, (s.get("adx_d", 20) - 10) / 2))  # map 10..30 → 0..10
        fund_score = s.get("total", 50) / 10  # 0..10
        prob_score = s["prob_6m"] / 10  # 0..9.5

        # Penalize negative alpha and low ADX
        if s.get("alpha_1m", 0) < -5:
            alpha_score *= 0.5
        if s.get("adx_d", 20) < 20:
            adx_score *= 0.5

        s["_pick_score"] = round(
            prob_score * 0.40 + alpha_score * 0.25 + fund_score * 0.20 + adx_score * 0.15, 2
        )

    portfolio.sort(key=lambda s: s["_pick_score"], reverse=True)
    portfolio = portfolio[:top_n]

    # Assign buy ranks
    for i, s in enumerate(portfolio):
        s["buy_rank"] = i + 1
        s["signal_strength"] = _signal_label(s)

    return portfolio


def _signal_label(s: Dict) -> str:
    """Human-readable signal strength label."""
    signals = 0
    if s["prob_6m"] >= 55: signals += 1
    if s.get("alpha_1m", 0) > 0: signals += 1
    if s.get("adx_d", 20) > 25: signals += 1
    if s.get("total", 50) >= 60: signals += 1
    if s.get("mtf_resonance", 0) >= 2: signals += 1

    if signals >= 4: return "🟢 强信号"
    if signals >= 3: return "🟡 中等"
    if signals >= 2: return "🟠 偏弱"
    return "🔴 仅有"


# ── Scoring probability color ────────────────────────────────────────────

def prob_color(v):
    if v >= 70: return "green"
    if v >= 50: return "gold"
    return "red"


# ── Main (CLI test) ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="A股多周期趋势预判")
    p.add_argument("--max", type=int, default=50, help="max stocks")
    p.add_argument("--top", type=int, default=20, help="top N output")
    p.add_argument("--rate", type=float, default=5.0, help="queries/sec")
    p.add_argument("--sort", type=str, default="prob_6m",
                   choices=["prob_1m", "prob_2m", "prob_3m", "prob_6m"],
                   help="sort by which horizon")
    args = p.parse_args()

    results, stats, portfolio = predict_batch(
        max_stocks=args.max, top_n=args.top,
        rate_limit=args.rate, horizon_sort=args.sort,
    )

    print(f"\n{'='*100}")
    print(f"  📊 推荐买入组合 ({len(portfolio)} 支)")
    print(f"{'='*100}")
    print(f"{'排名':>3}  {'代码':<9} {'名称':<8} {'半年↑':>6} {'Alpha':>6} {'ADX':>5} {'总分':>5} {'信号':<10} {'行业'}")
    print("-" * 100)
    for s in portfolio:
        print(f"{s['buy_rank']:>3}  {s['code']:<9} {s['name']:<8} "
              f"{s['prob_6m']:>5.0f}% {s.get('alpha_1m',0):>+5.1f}% {s.get('adx_d',0):>5.1f} "
              f"{s.get('total',0):>5.1f} {s['signal_strength']:<10} {s.get('industry','?')}")

    print(f"\n{'='*100}")
    print(f"  趋势预判排名")
    print(f"{'='*100}")
    print(f"{'#':>3}  {'代码':<9} {'名称':<8} {'1月↑':>6} {'2月↑':>6} {'3月↑':>6} {'半年↑':>6}  "
          f"{'技术D':>5} {'技术W':>5} {'技术M':>5} {'财务':>5} {'ROE':>6} {'阶段':>3} {'共振':>3}")
    print("-" * 100)
    for i, r in enumerate(results):
        em = "🔴" if r["mtf_resonance"] >= 3 else ("🟡" if r["mtf_resonance"] >= 2 else "⚪")
        print(f"{i+1:>3}  {r['code']:<9} {r['name']:<8} "
              f"{r['prob_1m']:>5.0f}% {r['prob_2m']:>5.0f}% {r['prob_3m']:>5.0f}% {r['prob_6m']:>5.0f}%  "
              f"{r['tech_d']:>5.0f} {r['tech_w']:>5.0f} {r['tech_m']:>5.0f} {r['fin_score']:>5.0f} "
              f"{r['roe']:>5.1f}% {r['stage_d']:>3}  {em}")

    print(f"\nPipeline: {stats['universe']} → T1:{stats['t1_passed']} → Predicted:{stats['predicted']} → Portfolio:{stats.get('portfolio_n',0)} | {stats['elapsed']}s")
