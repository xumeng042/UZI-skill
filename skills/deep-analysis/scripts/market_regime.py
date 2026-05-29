#!/usr/bin/env python3
"""Market regime detection and probability adjustment.

Uses 沪深300 (sh.000300) daily K-lines to classify market state
and applies calibrated probability multipliers to reduce bullish
bias in bear markets.

Regime classification:
- 强势 (Strong):  price > MA20 > MA60, MA20 sloping up
- 震荡 (Sideways): mixed signals, no clear trend
- 弱势 (Weak):    price < MA20 < MA60, MA20 sloping down
"""

import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

# ── Index data fetching ──────────────────────────────────────────────────

def _ma(values, n):
    if len(values) < n:
        return []
    out = []
    for i in range(len(values)):
        window = values[max(0, i - n + 1):i + 1]
        out.append(sum(window) / len(window))
    return out


def fetch_index_klines(bs, code: str = "sh.000300",
                       lookback_days: int = 500,
                       end_date: str = "") -> Optional[Dict]:
    """Fetch index K-lines for regime detection.

    Args:
        end_date: optional YYYY-MM-DD cutoff (default: today).
    """
    end = end_date or time.strftime("%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d") if end_date else datetime.now()
    start = (end_dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    rs = bs.query_history_k_data_plus(
        code, "date,open,high,low,close,volume",
        start_date=start, end_date=end,
        frequency="d", adjustflag="2",
    )
    if rs.error_code != "0":
        return None

    closes, highs, lows, volumes = [], [], [], []
    while rs.next():
        row = rs.get_row_data()
        try:
            closes.append(float(row[4]) if row[4] else 0)
            highs.append(float(row[2]) if row[2] else 0)
            lows.append(float(row[3]) if row[3] else 0)
            volumes.append(float(row[5]) if row[5] else 0)
        except (ValueError, IndexError):
            continue

    return {"closes": closes, "highs": highs, "lows": lows, "volumes": volumes}


# ── Regime classification ───────────────────────────────────────────────

def classify_regime(klines: Dict) -> Dict:
    """Classify current market regime from index K-lines.

    Returns dict with regime, confidence, and probability multiplier.
    """
    closes = klines["closes"]
    if len(closes) < 120:
        return {"regime": "未知", "confidence": 0, "prob_multiplier": 1.0,
                "score": 50, "signals": []}

    ma20 = _ma(closes, 20)
    ma60 = _ma(closes, 60)
    ma200 = _ma(closes, 200) if len(closes) >= 200 else []

    last = closes[-1]
    ma20_now = ma20[-1] if ma20 else last
    ma60_now = ma60[-1] if ma60 else last
    ma200_now = ma200[-1] if ma200 else last

    signals = []
    score = 50

    # 1. Price vs key MAs
    above_ma20 = last > ma20_now
    above_ma60 = last > ma60_now
    above_ma200 = last > ma200_now

    if above_ma20 and above_ma60 and above_ma200:
        score += 20
        signals.append("站上全部均线")
    elif above_ma20 and above_ma60:
        score += 10
        signals.append("站上MA20/60")
    elif not above_ma20 and not above_ma60:
        score -= 20
        signals.append("跌破MA20/60")
    elif not above_ma20:
        score -= 10
        signals.append("跌破MA20")

    # 2. MA slopes
    if len(ma20) >= 10:
        ma20_slope = (ma20_now - ma20[-10]) / ma20[-10] * 100 if ma20[-10] > 0 else 0
        if ma20_slope > 1:
            score += 10
            signals.append("MA20加速上行")
        elif ma20_slope > 0:
            score += 5
        elif ma20_slope < -1:
            score -= 10
            signals.append("MA20加速下行")
        elif ma20_slope < 0:
            score -= 5

    if len(ma60) >= 20:
        ma60_slope = (ma60_now - ma60[-20]) / ma60[-20] * 100 if ma60[-20] > 0 else 0
        if ma60_slope > 0.5:
            score += 5
        elif ma60_slope < -0.5:
            score -= 5

    # 3. Recent drawdown
    n_lookback = min(60, len(closes))
    recent_high = max(closes[-n_lookback:])
    drawdown = (last - recent_high) / recent_high * 100 if recent_high > 0 else 0
    if drawdown < -10:
        score -= 15
        signals.append(f"近期回撤{abs(drawdown):.0f}%")
    elif drawdown < -5:
        score -= 5

    # 4. Volume trend
    vols = klines["volumes"]
    if len(vols) >= 20:
        avg_vol_5 = sum(vols[-5:]) / 5 if vols[-5:] else 1
        avg_vol_20 = sum(vols[-20:]) / 20 if vols[-20:] else 1
        vol_ratio = avg_vol_5 / avg_vol_20 if avg_vol_20 > 0 else 1
        if vol_ratio > 1.3:
            score += 5
            signals.append("放量")
        elif vol_ratio < 0.7:
            score -= 5
            signals.append("缩量")

    # 5. Regime determination
    if score >= 65:
        regime = "强势"
        prob_multiplier = 1.0  # no adjustment needed
        confidence = min(100, score)
    elif score >= 40:
        regime = "震荡"
        # In sideways: slightly dampen extremes
        prob_multiplier = 0.95
        confidence = score
    else:
        regime = "弱势"
        # In bear: aggressive dampening
        # The deeper into bear territory, the stronger the dampening
        bear_depth = max(0, 40 - score)  # 0 to 40+
        prob_multiplier = max(0.6, 1.0 - bear_depth * 0.01)
        confidence = max(0, score)

    return {
        "regime": regime,
        "score": score,
        "confidence": confidence,
        "prob_multiplier": round(prob_multiplier, 3),
        "signals": signals,
        "index_last": round(last, 2),
        "ma20": round(ma20_now, 2),
        "ma60": round(ma60_now, 2),
        "drawdown_pct": round(drawdown, 1),
        # Index returns for RS (Relative Strength) computation
        "ret_1m": round((closes[-1] / closes[-min(20, len(closes))] - 1) * 100, 1) if len(closes) >= 20 else 0,
        "ret_3m": round((closes[-1] / closes[-min(60, len(closes))] - 1) * 100, 1) if len(closes) >= 60 else 0,
    }


# ── Apply regime adjustment to probabilities ─────────────────────────────

def adjust_probabilities(probs: Dict[str, float],
                         regime: Dict) -> Dict[str, float]:
    """Apply market regime adjustment to predicted probabilities.

    In weak markets, all upside probabilities are scaled down.
    In strong markets, probabilities are left as-is.
    The adjustment is stronger for shorter horizons (more noise-sensitive).
    """
    multiplier = regime["prob_multiplier"]

    # Regime-specific adjustment per horizon
    if regime["regime"] == "弱势":
        # Short horizons are more affected by bear trend
        adjusted = {
            "prob_1m": probs["prob_1m"] * (multiplier - 0.05),
            "prob_2m": probs["prob_2m"] * (multiplier - 0.03),
            "prob_3m": probs["prob_3m"] * multiplier,
            "prob_6m": probs["prob_6m"] * (multiplier + 0.02),
        }
    elif regime["regime"] == "震荡":
        adjusted = {k: v * multiplier for k, v in probs.items()}
    else:
        # 强势: slight boost for longer horizons
        adjusted = {
            "prob_1m": probs["prob_1m"],
            "prob_2m": probs["prob_2m"],
            "prob_3m": probs["prob_3m"] * 1.02,
            "prob_6m": probs["prob_6m"] * 1.03,
        }

    # Clamp
    return {k: round(min(95, max(5, v)), 1) for k, v in adjusted.items()}


# ── Cached singleton for batch use ───────────────────────────────────────

_regime_cache: Optional[Dict] = None
_cache_ts: float = 0
_CACHE_TTL = 600  # 10 minutes


def get_regime(bs) -> Dict:
    """Get current market regime with caching."""
    global _regime_cache, _cache_ts
    now = time.monotonic()
    if _regime_cache is not None and (now - _cache_ts) < _CACHE_TTL:
        return _regime_cache

    klines = fetch_index_klines(bs)
    if klines is None:
        return {"regime": "未知", "confidence": 0, "prob_multiplier": 1.0,
                "score": 50, "signals": ["无法获取指数数据"]}

    _regime_cache = classify_regime(klines)
    _cache_ts = now
    return _regime_cache


# ── CLI test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import baostock as bs
    bs.login()
    try:
        regime = get_regime(bs)
        print(f"Regime: {regime['regime']}")
        print(f"  Score: {regime['score']}")
        print(f"  Multiplier: {regime['prob_multiplier']}")
        print(f"  Signals: {regime['signals']}")
        print(f"  Index: {regime['index_last']} (MA20={regime['ma20']}, MA60={regime['ma60']})")
        print(f"  Drawdown: {regime['drawdown_pct']}%")

        # Test adjustment
        test_probs = {"prob_1m": 65, "prob_2m": 60, "prob_3m": 55, "prob_6m": 58}
        adj = adjust_probabilities(test_probs, regime)
        print(f"\n  Original: {test_probs}")
        print(f"  Adjusted: {adj}")
    finally:
        bs.logout()
