#!/usr/bin/env python3
"""分红融资比选股模块 — 基于"分红融资比>1"金标准.

核心理念（来自知乎 kaer）:
- 一个企业的历史累计分红必须 > 历史融资额，才值得投资
- 分红融资比 > 1 说明公司赚了钱且愿意分给股东，管理层德才兼备
- A股约 800/5000 只满足此条件

筛选逻辑:
1. 基础层: 分红融资比 > 1 (核心金标准)
2. 质量层: 分红比例 30%-70%
3. 回报层: 当前股息率
4. 行业层: 排除衰退行业
5. 承诺层: 分红承诺/章程
"""

import sys
import os
import time
from typing import Optional, Dict, List, Tuple

import baostock as bs
import pandas as pd
import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, os.path.join(SCRIPTS_DIR, "lib"))

from screen_stocks import (
    fetch_universe, filter_basic,
    fetch_one_stock,
)

# ── 衰退行业 / 夕阳行业 ──────────────────────────────────────────────────
DECLINING_INDUSTRIES = {
    "煤炭", "钢铁", "纺织", "造纸", "印刷", "化纤",
    "水泥", "玻璃", "焦炭", "普通钢铁", "普钢", "特钢",
    "火力发电", "煤化工",
}

LOW_GROWTH_INDUSTRIES = {
    "房地产", "房地产开发", "建筑", "建筑施工", "建材",
    "高速公路", "港口", "机场",
    "传统零售", "百货", "超市",
    "报纸", "出版", "广播电视",
    "固定电话", "传呼",
}


def _is_declining_industry(industry: str) -> bool:
    """Check if industry is declining or low-growth."""
    if not industry:
        return False
    for kw in DECLINING_INDUSTRIES | LOW_GROWTH_INDUSTRIES:
        if kw in industry:
            return True
    return False


# ── 分红融资比计算 ───────────────────────────────────────────────────────

def fetch_dividend_history(code: str) -> List[Dict]:
    """Fetch complete dividend history for a stock using baostock.

    Returns list of dicts with keys:
      year, cash_ps (dividend per share), stock_ps (stock dividend per share),
      total_shares_yi (total shares in 亿)
    """
    dividends = []
    try:
        for year in range(2000, 2026):
            rs = bs.query_dividend_data(code=code, year=str(year), yearType="report")
            if rs.error_code != "0":
                continue
            df = rs.get_data()
            if df.empty:
                continue
            for _, row in df.iterrows():
                cash_ps = _safe_float(row.get("dividCashPsBeforeTax", 0))
                stock_ps = _safe_float(row.get("dividStocksPs", 0))
                if cash_ps > 0 or stock_ps > 0:
                    dividends.append({
                        "year": year,
                        "cash_ps": cash_ps,
                        "stock_ps": stock_ps,
                        "desc": str(row.get("dividCashStock", "")),
                    })
    except Exception:
        pass
    return dividends


def fetch_dividend_history_em(code: str) -> List[Dict]:
    """Fetch dividend history from akshare (Eastmoney) — more detailed."""
    dividends = []
    try:
        import akshare as ak
        df = ak.stock_fhps_detail_em(symbol=code)
        if df is None or df.empty:
            return dividends
        for _, row in df.iterrows():
            report_date = str(row.get("报告期", ""))
            cash_ratio = _safe_float(row.get("现金分红-现金分红比例", 0))
            div_yield = _safe_float(row.get("现金分红-股息率", 0))
            eps = _safe_float(row.get("每股收益", 0))
            progress = str(row.get("方案进度", ""))
            if progress != "实施分配":
                continue
            if cash_ratio > 0:
                dividends.append({
                    "year": report_date[:4] if report_date else "",
                    "cash_per_10": cash_ratio,
                    "cash_ps": round(cash_ratio / 10, 4),
                    "div_yield": div_yield,
                    "eps": eps,
                })
    except Exception:
        pass
    return dividends


def _safe_float(val) -> float:
    """Safely convert value to float."""
    try:
        if val is None or val == "":
            return 0.0
        return float(str(val).replace(",", "").replace("%", ""))
    except (ValueError, TypeError):
        return 0.0


def calc_dividend_financing_ratio(m: dict, bs_mod) -> dict:
    """Calculate dividend-to-financing ratio and related metrics.

    Returns dict with:
      div_fin_ratio: 分红融资比
      total_div_per_share: 累计每股分红
      total_financing_per_share_est: 估算累计每股融资
      consecutive_div_years: 连续分红年数
      total_div_years: 总分红年数
      avg_div_yield_3y: 近3年平均股息率
      payout_ratio: 分红比例(近一年)
    """
    code = m.get("code", "")
    # Convert SH600000 → sh.600000 for baostock
    bs_code = code.replace("SH", "sh.").replace("SZ", "sz.").replace("BJ", "bj.")
    # Also handle SH.600000 format (dot variant)
    bs_code = bs_code.replace("sh..", "sh.").replace("sz..", "sz.").replace("bj..", "bj.")

    result = {
        "div_fin_ratio": 0.0,
        "total_div_per_share": 0.0,
        "total_financing_per_share_est": 0.0,
        "consecutive_div_years": 0,
        "total_div_years": 0,
        "avg_div_yield_3y": 0.0,
        "payout_ratio": 0.0,
        "ipo_price": 0.0,
        "dividend_records": [],
        "current_price": 0.0,
    }

    # ── 0. Fetch current price ──
    current_price = _fetch_current_price(bs_mod, bs_code)
    result["current_price"] = current_price

    # ── 1. Fetch dividend history ──
    div_records = fetch_dividend_history(bs_code)
    result["dividend_records"] = div_records

    result["dividend_records"] = div_records

    # ── 2. Calculate cumulative dividend per share ──
    total_cash_ps = sum(r["cash_ps"] for r in div_records)
    result["total_div_per_share"] = round(total_cash_ps, 2)

    # ── 3. Count dividend years and consecutive years ──
    div_years_set = set()
    for r in div_records:
        y = r.get("year", "")
        if y is not None:
            if isinstance(y, int):
                div_years_set.add(y)
            elif isinstance(y, str) and y.isdigit():
                div_years_set.add(int(y))

    result["total_div_years"] = len(div_years_set)

    # Consecutive years (going backwards from most recent)
    sorted_years = sorted(div_years_set, reverse=True)
    consecutive = 0
    for i, y in enumerate(sorted_years):
        if i == 0:
            consecutive = 1
        elif sorted_years[i-1] - y == 1:
            consecutive += 1
        else:
            break
    result["consecutive_div_years"] = consecutive

    # ── 4. Average dividend yield (last 3 years) ──
    recent_divs = sorted(div_records, key=lambda r: str(r.get("year", "")), reverse=True)
    # Compute yield from dividend per share / current price
    price = current_price
    if price > 0:
        yields_3y = []
        for r in recent_divs[:3]:
            cash_ps = r.get("cash_ps", 0)
            if cash_ps > 0:
                yields_3y.append(round(cash_ps / price * 100, 2))
        if yields_3y:
            result["avg_div_yield_3y"] = round(sum(yields_3y) / len(yields_3y), 2)

    # ── 5. Estimate financing (IPO + refinancing) ──
    ipo_price = _estimate_ipo_financing(bs_code)
    result["ipo_price"] = ipo_price
    total_financing_ps = ipo_price
    result["total_financing_per_share_est"] = round(total_financing_ps, 2)

    # ── 6. Dividend-to-financing ratio ──
    if total_financing_ps > 0:
        result["div_fin_ratio"] = round(total_cash_ps / total_financing_ps, 2)
    elif total_cash_ps > 0:
        # If we can't estimate financing but company pays dividends, assume good
        result["div_fin_ratio"] = 999.0

    # ── 7. Payout ratio ──
    # Estimate from latest dividend per share / current_price and net_profit
    latest_div_ps = recent_divs[0]["cash_ps"] if recent_divs else 0
    # Approximate shares from net_profit_latest (亿) and earnings per share estimate
    # Use simple approach: payout_ratio = latest_dividend_ps / (net_profit / shares)
    # We don't have shares, so estimate from price * shares ≈ mcap if available
    # Fallback: use industry-average PE to estimate EPS from price
    net_profit_latest = m.get("net_profit_latest", 0) or 0
    if price > 0 and latest_div_ps > 0 and net_profit_latest > 0:
        # Estimate total shares from net profit and a reasonable PE
        # Assume P/E ~15, so price / 15 = EPS, shares = net_profit / EPS
        est_eps = price / 15.0
        if est_eps > 0:
            est_shares_yi = net_profit_latest / est_eps  # in 亿
            eps = net_profit_latest / est_shares_yi if est_shares_yi > 0 else 0
            if eps > 0:
                result["payout_ratio"] = round(latest_div_ps / eps * 100, 1)

    return result


def _fetch_current_price(bs_mod, bs_code: str) -> float:
    """Fetch current stock price from latest daily kline."""
    try:
        rs = bs_mod.query_history_k_data_plus(
            bs_code, "date,close",
            start_date="2026-01-01", end_date="2026-12-31",
            frequency="d", adjustflag="3",
        )
        if rs.error_code != "0":
            return 0.0
        df = rs.get_data()
        if df.empty:
            return 0.0
        return _safe_float(df.iloc[-1].get("close", 0))
    except Exception:
        return 0.0


def _estimate_ipo_financing(bs_code: str) -> float:
    """Estimate IPO price for a stock. Returns per-share financing amount.

    Uses earliest available monthly close as a rough IPO price estimate.
    Falls back to reasonable defaults when data is unavailable.
    """
    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,close",
            start_date="1990-01-01",
            end_date="2026-12-31",
            frequency="m",
            adjustflag="3",  # 不复权：最接近实际IPO价格
        )
        if rs.error_code != "0":
            return 0.0
        df = rs.get_data()
        if df.empty:
            return 0.0
        first_row = df.iloc[0]
        val = _safe_float(first_row.get("close", 0))
        if val <= 0:
            val = _safe_float(first_row.get("open", 0))
        return val
    except Exception:
        return 0.0


# ── Dividend commitment check ──

def check_dividend_commitment(code: str, name: str) -> dict:
    """Check if the company has dividend commitments in its charter.

    Returns dict with:
      has_commitment: bool
      commitment_desc: str
      confidence: 'high' | 'medium' | 'low'
    """
    result = {
        "has_commitment": False,
        "commitment_desc": "",
        "confidence": "low",
    }

    # Known high-dividend-commitment companies (常见分红承诺蓝筹)
    HIGH_COMMITMENT = {
        "600036": ("招商银行", "公司章程规定分红比例不低于30%"),
        "601398": ("工商银行", "承诺分红比例不低于30%"),
        "601939": ("建设银行", "承诺分红比例不低于30%"),
        "601288": ("农业银行", "承诺分红比例不低于30%"),
        "601988": ("中国银行", "承诺分红比例不低于30%"),
        "601328": ("交通银行", "承诺分红比例不低于30%"),
        "600900": ("长江电力", "承诺分红比例不低于50%，2021-2025年≥70%"),
        "600519": ("贵州茅台", "持续高分红传统，分红率>50%"),
        "000858": ("五粮液", "承诺分红率不低于50%"),
        "600585": ("海螺水泥", "持续高分红，分红率>30%"),
        "601088": ("中国神华", "承诺分红率不低于50%"),
        "600028": ("中国石化", "承诺分红率不低于30%"),
        "601857": ("中国石油", "承诺分红率不低于30%"),
        "601166": ("兴业银行", "分红率持续>25%"),
        "000333": ("美的集团", "分红率持续>40%"),
        "000651": ("格力电器", "分红率持续>50%"),
        "600887": ("伊利股份", "分红率持续>50%"),
    }

    clean_code = code.replace("SH.", "").replace("SZ.", "").replace("BJ.", "")
    if clean_code in HIGH_COMMITMENT:
        expected_name, desc = HIGH_COMMITMENT[clean_code]
        result["has_commitment"] = True
        result["commitment_desc"] = desc
        result["confidence"] = "high"
    elif name and clean_code in {k: v[0] for k, v in HIGH_COMMITMENT.items()}:
        result["has_commitment"] = True
        result["commitment_desc"] = HIGH_COMMITMENT.get(clean_code, ("", ""))[1]
        result["confidence"] = "medium"

    return result


# ── 综合评分 ─────────────────────────────────────────────────────────────

def score_dividend_stock(m: dict, div_data: dict, commitment: dict) -> dict:
    """Score a stock based on the article's methodology (0-100 scale).

    Scoring dimensions:
    - 分红融资比 (35分): 核心金标准，越高越好
    - 股息回报 (25分): 当前股息率
    - 分红质量 (20分): 连续分红年数 + 分红比例合理性
    - 基本面 (10分): ROE + 负债率
    - 分红承诺 (10分): 是否有章程保证
    """
    name = m.get("name", "")
    code = m.get("code", "")

    # ── 1. 分红融资比 (35分) ──
    ratio = div_data.get("div_fin_ratio", 0)
    div_fin_score = 0
    if ratio >= 3:
        div_fin_score = 35
    elif ratio >= 2:
        div_fin_score = 30
    elif ratio >= 1.5:
        div_fin_score = 25
    elif ratio >= 1.0:
        div_fin_score = 20
    elif ratio >= 0.5:
        div_fin_score = 10
    else:
        div_fin_score = 0

    # ── 2. 股息回报 (25分) ──
    div_yield = div_data.get("avg_div_yield_3y", 0) or m.get("div_yield", 0) or 0
    yield_score = 0
    if div_yield >= 7:
        yield_score = 25
    elif div_yield >= 5:
        yield_score = 20
    elif div_yield >= 4:
        yield_score = 16
    elif div_yield >= 3:
        yield_score = 12
    elif div_yield >= 2:
        yield_score = 8
    else:
        yield_score = 2

    # ── 3. 分红质量 (20分) ──
    consec_years = div_data.get("consecutive_div_years", 0)
    payout = div_data.get("payout_ratio", 0)

    quality_score = 0
    # 连续分红年限
    if consec_years >= 10:
        quality_score += 8
    elif consec_years >= 5:
        quality_score += 5
    elif consec_years >= 3:
        quality_score += 2

    # 总分红年数
    total_div_years = div_data.get("total_div_years", 0)
    if total_div_years >= 15:
        quality_score += 4
    elif total_div_years >= 10:
        quality_score += 2

    # 分红比例合理性 (30%-70%最优)
    if 30 <= payout <= 70:
        quality_score += 8
    elif 20 <= payout <= 80:
        quality_score += 4
    elif payout > 80:
        quality_score += 0  # 过高可能是杀鸡取卵
    else:
        quality_score += 2  # 没有数据

    quality_score = min(20, quality_score)

    # ── 4. 基本面 (10分) ──
    roe = m.get("roe_latest", 0) or 0
    debt = m.get("debt_ratio", 50) or 50

    fund_score = 0
    if roe >= 15:
        fund_score += 5
    elif roe >= 10:
        fund_score += 3
    elif roe >= 5:
        fund_score += 1

    if debt < 30:
        fund_score += 5
    elif debt < 50:
        fund_score += 3
    elif debt < 70:
        fund_score += 1

    fund_score = min(10, fund_score)

    # ── 5. 分红承诺 (10分) ──
    commit_score = 0
    if commitment.get("has_commitment"):
        if commitment.get("confidence") == "high":
            commit_score = 10
        elif commitment.get("confidence") == "medium":
            commit_score = 6
        else:
            commit_score = 3
    elif consec_years >= 10:
        commit_score = 5  # 虽然没有明文承诺但长期分红
    elif consec_years >= 5:
        commit_score = 2

    commit_score = min(10, commit_score)

    # ── Total ──
    total = div_fin_score + yield_score + quality_score + fund_score + commit_score

    industry = m.get("industry", "") or ""
    r = {
        "code": code,
        "name": name,
        "total": total,
        "div_fin_score": div_fin_score,
        "yield_score": yield_score,
        "quality_score": quality_score,
        "fund_score": fund_score,
        "commit_score": commit_score,
        "div_fin_ratio": ratio,
        "total_div_ps": div_data.get("total_div_per_share", 0),
        "div_yield": div_yield,
        "consecutive_div_years": consec_years,
        "total_div_years": div_data.get("total_div_years", 0),
        "payout_ratio": payout,
        "roe": roe,
        "debt": debt,
        "industry": industry,
        "is_declining": _is_declining_industry(industry),
        "has_commitment": commitment.get("has_commitment", False),
        "commitment_desc": commitment.get("commitment_desc", ""),
        "name_clean": name,
    }

    return r


# ── 主筛选管道 ──────────────────────────────────────────────────────────

def run_dividend_screening(max_stocks: int = 0, top_n: int = 30,
                            rate_limit: float = 3.0,
                            require_ratio_gt_1: bool = True,
                            require_payout_30_70: bool = True,
                            exclude_declining: bool = True,
                            min_div_yield: float = 2.0) -> Tuple[List[Dict], Dict]:
    """Run the dividend-to-financing ratio screening pipeline.

    Args:
        max_stocks: Max stocks to scan (0 = all)
        top_n: Top N results
        rate_limit: Queries per second
        require_ratio_gt_1: Only include stocks with ratio > 1
        require_payout_30_70: Only include stocks with payout 30-70%
        exclude_declining: Exclude declining industries
        min_div_yield: Minimum dividend yield %
    """
    t0 = time.monotonic()
    print("Fetching A-share universe for dividend screening...", file=sys.stderr, flush=True)
    df = fetch_universe()

    candidates = []
    for _, row in df.iterrows():
        code = str(row.iloc[0]).strip()
        name = str(row.iloc[1]).strip()
        if filter_basic(code, name):
            candidates.append((code, name))

    stats = {
        "universe": len(df),
        "t1_passed": len(candidates),
    }

    if max_stocks > 0:
        candidates = candidates[:max_stocks]

    results = []
    failures = 0
    passing_ratio = 0  # count of stocks with ratio > 1

    bs.login()
    try:
        for i, (code, name) in enumerate(candidates):
            if (i + 1) % 50 == 0:
                elapsed = time.monotonic() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  {i+1}/{len(candidates)} ({rate:.1f}/s) | "
                      f"{len(results)} scored | {passing_ratio} ratio>1",
                      file=sys.stderr, flush=True)

            time.sleep(1.0 / rate_limit)

            # Fetch financial data
            m = fetch_one_stock(code, name)
            if m is None:
                failures += 1
                continue

            # Calculate dividend-to-financing ratio
            div_data = calc_dividend_financing_ratio(m, bs)
            ratio = div_data.get("div_fin_ratio", 0)

            # Count stocks passing the golden rule
            if ratio > 1:
                passing_ratio += 1

            # Filter: ratio must be > 1 (golden rule)
            if require_ratio_gt_1 and ratio <= 1:
                continue

            # Filter: payout ratio in 30-70%
            payout = div_data.get("payout_ratio", 0)
            if require_payout_30_70 and (payout < 30 or payout > 70):
                # Still include if ratio is very high (> 3) - exceptional companies
                if ratio < 3:
                    continue

            # Filter: minimum dividend yield
            avg_yield = div_data.get("avg_div_yield_3y", 0) or m.get("div_yield", 0) or 0
            if avg_yield < min_div_yield:
                continue

            # Check dividend commitment
            commitment = check_dividend_commitment(code, name)

            # Filter: exclude declining industries
            industry = m.get("industry", "") or ""
            if exclude_declining and _is_declining_industry(industry):
                continue

            # Score
            scored = score_dividend_stock(m, div_data, commitment)
            results.append(scored)
    finally:
        bs.logout()

    # Sort by total score descending
    results.sort(key=lambda r: r["total"], reverse=True)
    results = results[:top_n]

    elapsed = time.monotonic() - t0
    stats.update({
        "elapsed": round(elapsed, 1),
        "passed": len(results),
        "failures": failures,
        "ratio_gt_1_count": passing_ratio,
        "ratio_gt_1_pct": round(passing_ratio / max(len(candidates), 1) * 100, 1),
    })

    return results, stats


# ── CLI test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="分红融资比选股")
    p.add_argument("--max", type=int, default=200, help="max stocks to scan")
    p.add_argument("--top", type=int, default=30, help="top N output")
    p.add_argument("--rate", type=float, default=3.0, help="queries/sec")
    p.add_argument("--all", action="store_true", help="include all (even ratio <= 1)")
    args = p.parse_args()

    results, stats = run_dividend_screening(
        max_stocks=args.max,
        top_n=args.top,
        rate_limit=args.rate,
        require_ratio_gt_1=not args.all,
    )

    print(f"\n{'='*110}")
    print(f"  💎 分红融资比选股 (金标准: 历史分红 > 历史融资)")
    print(f"{'='*110}")
    print(f"A股全量: {stats['universe']} → "
          f"T1基础: {stats['t1_passed']} → "
          f"分红融资比>1: {stats['ratio_gt_1_count']} ({stats['ratio_gt_1_pct']}%) → "
          f"最终入选: {stats['passed']} | {stats['elapsed']}s")
    print(f"{'='*110}")
    print(f"{'#':>3} {'代码':<9} {'名称':<8} {'总分':>5} {'融资比':>6} {'股息':>5} "
          f"{'连分':>4} {'分红%':>5} {'ROE':>5} {'负债':>5} {'行业':<10} {'承诺'}")
    print("-" * 110)

    for i, r in enumerate(results):
        print(f"{i+1:>3} {r['code']:<9} {r['name']:<8} {r['total']:>5.1f} "
              f"{r['div_fin_ratio']:>5.1f}x {r['div_yield']:>4.1f}% "
              f"{r['consecutive_div_years']:>4} {r['payout_ratio']:>4.0f}% "
              f"{r['roe']:>4.1f}% {r['debt']:>4.1f}% "
              f"{r['industry'][:10]:<10} {'✓' if r['has_commitment'] else '-'}")
