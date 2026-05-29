#!/usr/bin/env python3
"""Industry-relative benchmark and z-score normalization.

Provides industry median/quartile benchmarks for key financial metrics,
enabling within-industry relative scoring instead of absolute thresholds.

Two modes:
1. Static lookup table (fast, always available)
2. Dynamic computation from batch data (more accurate, used in screening)
"""

from statistics import mean, stdev, median
from typing import Dict, List, Optional, Tuple


# ── Static industry benchmarks ───────────────────────────────────────────
# (roe_median, roe_std, margin_median, margin_std, debt_median, debt_std,
#  growth_median, growth_std)
# Based on approximate A-share sector averages.

_STATIC_BENCHMARKS: Dict[str, Tuple[float, float, float, float, float, float, float, float]] = {
    # Code pattern → (roe, roe_std, net_margin, margin_std, debt, debt_std, rev_growth, growth_std)
    "C15": (20, 10, 25, 12, 30, 12, 12, 10),   # 酒、饮料
    "C14": (12, 6, 10, 5, 35, 12, 8, 6),        # 食品制造
    "C13": (8, 5, 5, 3, 40, 12, 6, 5),          # 农副食品
    "C27": (12, 8, 15, 10, 30, 12, 10, 8),      # 医药制造
    "C26": (10, 7, 8, 5, 40, 12, 8, 7),         # 化学原料
    "C30": (10, 6, 8, 5, 42, 12, 7, 6),         # 非金属矿物
    "C32": (8, 7, 5, 4, 45, 12, 8, 7),          # 有色金属
    "C33": (8, 6, 5, 3, 45, 12, 6, 5),          # 金属制品
    "C34": (8, 5, 8, 5, 38, 12, 7, 6),          # 通用设备
    "C35": (9, 6, 10, 6, 35, 12, 10, 8),        # 专用设备
    "C36": (10, 7, 6, 4, 50, 12, 8, 7),         # 汽车制造
    "C37": (6, 5, 5, 3, 50, 12, 8, 7),          # 铁路船舶航空航天
    "C38": (10, 7, 8, 5, 45, 12, 12, 10),       # 电气机械（新能源）
    "C39": (8, 7, 8, 6, 35, 12, 15, 12),        # 计算机通信电子
    "C40": (8, 5, 10, 6, 30, 12, 10, 8),        # 仪器仪表
    "D44": (8, 4, 10, 5, 60, 12, 6, 5),         # 电力热力
    "D45": (8, 4, 8, 4, 55, 12, 8, 6),          # 燃气
    "E47": (8, 5, 6, 3, 65, 12, 5, 5),          # 房屋建筑
    "E48": (8, 4, 5, 3, 65, 10, 8, 7),          # 土木工程
    "F51": (8, 5, 3, 2, 45, 12, 8, 7),          # 批发
    "F52": (8, 5, 4, 3, 50, 12, 7, 6),          # 零售
    "G54": (6, 4, 15, 8, 40, 12, 6, 5),         # 道路运输
    "G56": (5, 5, 10, 8, 55, 12, 5, 5),         # 航空运输
    "I63": (6, 4, 10, 5, 30, 12, 8, 7),         # 电信
    "I64": (6, 4, 8, 5, 30, 12, 10, 8),         # 互联网
    "I65": (8, 6, 10, 7, 25, 10, 15, 12),       # 软件
    "J66": (10, 2, 30, 5, 75, 5, 8, 3),         # 银行（低方差）
    "J67": (8, 4, 25, 8, 60, 10, 10, 8),        # 券商
    "J68": (10, 3, 20, 5, 55, 8, 8, 5),         # 保险
    "K70": (6, 4, 8, 5, 60, 12, 5, 5),          # 房地产
    "L72": (10, 6, 8, 5, 40, 12, 10, 8),        # 商务服务
    "M73": (10, 6, 12, 8, 25, 10, 12, 10),      # 科研
    "N77": (8, 4, 10, 5, 45, 12, 10, 8),        # 生态环保
    "Q84": (10, 5, 8, 5, 30, 12, 8, 7),         # 卫生
    "R85": (8, 5, 10, 6, 30, 12, 8, 6),         # 新闻出版
    "R86": (8, 5, 12, 6, 25, 10, 10, 8),        # 广播影视
    "A01": (5, 4, 5, 3, 40, 12, 5, 5),          # 农业
    "A02": (5, 4, 8, 5, 35, 12, 6, 5),          # 林业
    "B06": (10, 6, 10, 6, 35, 12, 6, 5),        # 煤炭
    "B07": (8, 5, 8, 5, 35, 12, 5, 5),          # 石油
    "B08": (8, 5, 8, 5, 30, 12, 8, 6),          # 黑色金属
    "B09": (8, 6, 8, 5, 35, 12, 8, 7),          # 有色金属矿
    "B10": (8, 5, 10, 5, 25, 10, 6, 5),         # 非金属矿
    "B11": (10, 6, 15, 8, 35, 12, 8, 7),        # 开采辅助
    "C17": (8, 4, 8, 4, 35, 12, 5, 4),          # 纺织
    "C18": (8, 5, 8, 4, 30, 12, 6, 5),          # 服装
    "C19": (8, 5, 8, 4, 30, 12, 6, 5),          # 皮革
    "C20": (8, 5, 8, 4, 35, 12, 5, 5),          # 木材
    "C21": (8, 5, 8, 4, 35, 12, 6, 5),          # 家具
    "C22": (8, 5, 8, 4, 35, 12, 6, 5),          # 造纸
    "C23": (8, 5, 8, 5, 30, 12, 8, 7),          # 印刷
    "C24": (10, 5, 8, 4, 30, 12, 8, 7),         # 文体用品
    "C25": (8, 5, 8, 4, 35, 12, 6, 5),          # 石油加工
    "C28": (8, 5, 8, 4, 35, 12, 8, 6),          # 化学纤维
    "C29": (8, 5, 8, 4, 35, 12, 8, 6),          # 橡胶塑料
    "C31": (8, 5, 8, 4, 40, 12, 6, 5),          # 黑色金属加工
    "D46": (8, 4, 8, 4, 55, 10, 8, 6),          # 水生产供应
    "E49": (8, 5, 6, 3, 55, 12, 8, 7),          # 建筑安装
    "E50": (8, 5, 8, 5, 50, 12, 8, 7),          # 建筑装饰
    "G53": (6, 4, 8, 4, 40, 12, 6, 5),          # 铁路运输
    "G55": (6, 4, 10, 5, 40, 12, 5, 5),         # 水上运输
    "G57": (8, 4, 8, 4, 45, 12, 8, 6),          # 管道运输
    "G58": (8, 5, 5, 3, 40, 12, 10, 8),         # 装卸搬运
    "G59": (8, 5, 8, 5, 40, 12, 10, 8),         # 仓储
    "G60": (8, 4, 8, 4, 40, 12, 6, 5),          # 邮政
    "H61": (5, 4, 5, 3, 40, 12, 5, 5),          # 住宿
    "H62": (5, 4, 8, 5, 40, 12, 5, 5),          # 餐饮
    "J69": (8, 4, 15, 6, 50, 12, 6, 5),         # 其他金融
    "L71": (6, 4, 8, 5, 40, 12, 6, 5),          # 租赁
    "M74": (8, 5, 8, 5, 30, 12, 8, 7),          # 专业技术
    "N78": (6, 4, 6, 3, 40, 12, 8, 7),          # 公共设施
    "O79": (6, 4, 6, 3, 45, 12, 5, 5),          # 居民服务
    "O80": (8, 5, 8, 5, 25, 10, 10, 8),         # 机动车电子
    "O81": (6, 4, 8, 4, 35, 12, 6, 5),          # 其他服务
    "P82": (8, 5, 8, 6, 30, 12, 8, 7),          # 教育
    "Q83": (8, 5, 8, 5, 30, 12, 8, 7),          # 社会工作
    "R87": (8, 5, 10, 6, 25, 10, 8, 7),         # 文化艺术
    "S88": (8, 5, 12, 8, 25, 10, 8, 7),         # 体育
    "S89": (8, 5, 10, 6, 25, 10, 6, 5),         # 娱乐
}


def get_industry_prefix(industry_code: str) -> str:
    """Extract industry prefix for benchmark lookup."""
    if not industry_code:
        return ""
    return industry_code[:3] if len(industry_code) >= 3 else industry_code


def get_benchmark(industry_code: str) -> dict:
    """Get benchmark values for a given industry code.

    Returns dict with median and std for key metrics.
    Falls back to broad sector averages if specific code not found.
    """
    prefix = get_industry_prefix(industry_code)
    bm = _STATIC_BENCHMARKS.get(prefix)

    if bm is None:
        # Fallback: use broad sector category
        if not industry_code:
            bm = (8, 6, 8, 5, 40, 12, 8, 7)
        elif industry_code[0] == "C":
            bm = (9, 6, 8, 5, 38, 12, 9, 8)   # 制造业通用
        elif industry_code[0] in ("A", "B"):
            bm = (8, 5, 8, 5, 35, 12, 6, 5)   # 农林牧渔/采矿业
        elif industry_code[0] == "D":
            bm = (8, 4, 9, 5, 55, 12, 7, 6)   # 电力燃气
        elif industry_code[0] == "E":
            bm = (8, 5, 6, 3, 60, 12, 7, 6)   # 建筑业
        elif industry_code[0] == "F":
            bm = (8, 5, 4, 3, 48, 12, 8, 7)   # 批发零售
        elif industry_code[0] == "G":
            bm = (6, 4, 10, 5, 42, 12, 6, 5)  # 交通运输
        elif industry_code[0] == "H":
            bm = (5, 4, 6, 4, 40, 12, 5, 5)   # 住宿餐饮
        elif industry_code[0] == "I":
            bm = (7, 5, 9, 6, 28, 11, 12, 10) # 信息技术
        elif industry_code[0] == "J":
            bm = (9, 3, 25, 6, 65, 8, 9, 5)   # 金融
        elif industry_code[0] == "K":
            bm = (6, 4, 8, 5, 60, 12, 5, 5)   # 房地产
        elif industry_code[0] == "L":
            bm = (10, 6, 8, 5, 40, 12, 10, 8) # 商务服务
        elif industry_code[0] == "M":
            bm = (9, 5, 10, 7, 28, 11, 10, 9) # 科研
        elif industry_code[0] == "N":
            bm = (7, 4, 8, 4, 43, 12, 9, 7)   # 水利环境
        elif industry_code[0] in ("O", "P", "Q", "R", "S"):
            bm = (8, 5, 10, 6, 30, 12, 8, 7)  # 服务业
        else:
            bm = (8, 6, 8, 5, 40, 12, 8, 7)

    return {
        "roe_median": bm[0], "roe_std": bm[1],
        "margin_median": bm[2], "margin_std": bm[3],
        "debt_median": bm[4], "debt_std": bm[5],
        "growth_median": bm[6], "growth_std": bm[7],
    }


# ── Z-score computation ──────────────────────────────────────────────────

def compute_z_score(value: float, median_val: float, std_val: float) -> float:
    """Compute z-score: how many std devs from industry median.

    Returns -3 to +3 range, clipped.
    """
    if std_val <= 0:
        return 0
    z = (value - median_val) / std_val
    return max(-3.0, min(3.0, z))


def z_to_score(z: float) -> float:
    """Convert z-score to a 0-10 scoring scale.

    z=0   → 5.0 (at industry median)
    z=+1  → 7.0 (1 std above)
    z=+2  → 8.5 (2 std above)
    z=+3  → 10.0 (top of industry)
    z=-1  → 3.0
    z=-2  → 1.5
    z=-3  → 0.0
    """
    if z >= 0:
        return min(10, 5 + z * 2.0)
    else:
        return max(0, 5 + z * 1.67)


def industry_adjust_score(raw_score: float, z: float, weight: float = 0.3) -> float:
    """Blend raw absolute score with industry-relative z-score.

    Args:
        raw_score: original absolute score (0-10)
        z: z-score relative to industry peers
        weight: how much to weight industry-relative vs absolute (0-1)
    """
    rel_score = z_to_score(z)
    return raw_score * (1 - weight) + rel_score * weight


def score_with_industry_benchmarks(m: dict) -> dict:
    """Score a stock's key metrics relative to its industry peers.

    Returns dict with z-scores and adjusted scores for key dimensions.
    This is a supplement to score_stock(), not a replacement.
    """
    industry = m.get("industry", "")
    bm = get_benchmark(industry)

    # Compute z-scores for each metric
    roe = m.get("roe_latest", 0) or 0
    margin = m.get("net_margin", 0) or 0
    debt = m.get("debt_ratio", 50) or 50
    growth = m.get("revenue_cagr_3y", 0) or 0

    z_roe = compute_z_score(roe, bm["roe_median"], bm["roe_std"])
    z_margin = compute_z_score(margin, bm["margin_median"], bm["margin_std"])
    z_debt = compute_z_score(-debt, -bm["debt_median"], bm["debt_std"])  # negative: lower debt is better
    z_growth = compute_z_score(growth, bm["growth_median"], bm["growth_std"])

    return {
        "industry": industry,
        "industry_prefix": get_industry_prefix(industry),
        "benchmarks": bm,
        "z_roe": round(z_roe, 2),
        "z_margin": round(z_margin, 2),
        "z_debt": round(z_debt, 2),
        "z_growth": round(z_growth, 2),
        # Combined industry quality score (average of z-scores → 0-10)
        "industry_quality_score": round(
            (z_to_score(z_roe) + z_to_score(z_margin) + z_to_score(z_debt) + z_to_score(z_growth)) / 4, 1
        ),
    }


# ── Batch benchmark computation from live data ────────────────────────────

def compute_batch_benchmarks(stocks: List[dict]) -> Dict[str, dict]:
    """Compute industry benchmarks from a batch of stock data.

    For each industry group, computes median and std of key metrics.
    Used to get more accurate, up-to-date benchmarks during screening.
    """
    groups: Dict[str, List[dict]] = {}
    for m in stocks:
        prefix = get_industry_prefix(m.get("industry", ""))
        if not prefix:
            continue
        groups.setdefault(prefix, []).append(m)

    benchmarks = {}
    for prefix, members in groups.items():
        if len(members) < 3:
            benchmarks[prefix] = get_benchmark(prefix)
            continue

        roes = [m.get("roe_latest", 0) or 0 for m in members]
        margins = [m.get("net_margin", 0) or 0 for m in members]
        debts = [m.get("debt_ratio", 50) or 50 for m in members]
        growths = [m.get("revenue_cagr_3y", 0) or 0 for m in members]

        try:
            benchmarks[prefix] = {
                "roe_median": round(median(roes), 1),
                "roe_std": round(stdev(roes), 1) if len(roes) >= 5 else get_benchmark(prefix)["roe_std"],
                "margin_median": round(median(margins), 1),
                "margin_std": round(stdev(margins), 1) if len(margins) >= 5 else get_benchmark(prefix)["margin_std"],
                "debt_median": round(median(debts), 1),
                "debt_std": round(stdev(debts), 1) if len(debts) >= 5 else get_benchmark(prefix)["debt_std"],
                "growth_median": round(median(growths), 1),
                "growth_std": round(stdev(growths), 1) if len(growths) >= 5 else get_benchmark(prefix)["growth_std"],
                "_n": len(members),
            }
        except Exception:
            benchmarks[prefix] = get_benchmark(prefix)

    return benchmarks


# ── Industry-relative momentum ranking ──────────────────────────────────

def compute_relative_momentum(results: List[dict]) -> dict:
    """Compute within-industry momentum percentile for each stock.

    Groups stocks by industry prefix, ranks ret_1m and ret_3m within
    each group, and returns a mapping of code → momentum adjustment.

    Adjustment ranges from -3 (bottom of industry) to +3 (top of industry).
    Only applied to groups with >= 3 members.

    Args:
        results: list of prediction dicts, each with code, industry, ret_1m, ret_3m

    Returns:
        dict mapping code → {"ret_1m_pct": float, "ret_3m_pct": float,
                              "adj_1m": float, "adj_3m": float, "combined": float}
    """
    # Group by industry prefix
    groups: Dict[str, list] = {}
    for r in results:
        ind = r.get("industry", "")
        prefix = get_industry_prefix(ind) if ind else ""
        if not prefix:
            prefix = "_unknown"
        groups.setdefault(prefix, []).append(r)

    output = {}
    for prefix, members in groups.items():
        n = len(members)
        if n < 3:
            # Skip tiny groups — not enough data for relative ranking
            for r in members:
                output[r["code"]] = {
                    "ret_1m_pct": 50, "ret_3m_pct": 50,
                    "adj_1m": 0, "adj_3m": 0, "combined": 0,
                    "peer_count": n,
                }
            continue

        # Sort by ret_1m to compute percentile
        sorted_1m = sorted(members, key=lambda r: r.get("ret_1m", 0) or 0)
        sorted_3m = sorted(members, key=lambda r: r.get("ret_3m", 0) or 0)

        # Assign percentile ranks
        rank_1m = {r["code"]: i / (n - 1) * 100 for i, r in enumerate(sorted_1m)}
        rank_3m = {r["code"]: i / (n - 1) * 100 for i, r in enumerate(sorted_3m)}

        for r in members:
            code = r["code"]
            pct_1m = rank_1m[code]
            pct_3m = rank_3m[code]

            # Map percentile to adjustment
            def pct_to_adj(pct: float) -> float:
                if pct >= 80:
                    return 3.0   # top 20%
                elif pct >= 60:
                    return 1.5   # top 40%
                elif pct >= 40:
                    return 0     # middle
                elif pct >= 20:
                    return -1.5  # bottom 40%
                else:
                    return -3.0  # bottom 20%

            adj_1m = pct_to_adj(pct_1m)
            adj_3m = pct_to_adj(pct_3m)
            # Combined: weight 3m more (longer trend more meaningful)
            combined = round(adj_1m * 0.4 + adj_3m * 0.6, 1)

            output[code] = {
                "ret_1m_pct": round(pct_1m, 1),
                "ret_3m_pct": round(pct_3m, 1),
                "adj_1m": adj_1m,
                "adj_3m": adj_3m,
                "combined": combined,
                "peer_count": n,
            }

    return output


# ── CLI test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Static Benchmarks ===")
    test_codes = ["C15", "C27", "C39", "J66", "D44", "I65", "C36", "A01", "K70"]
    for code in test_codes:
        bm = get_benchmark(code)
        print(f"  {code}: ROE~{bm['roe_median']}±{bm['roe_std']}  "
              f"Margin~{bm['margin_median']}±{bm['margin_std']}  "
              f"Debt~{bm['debt_median']}±{bm['debt_std']}  "
              f"Growth~{bm['growth_median']}±{bm['growth_std']}")

    print("\n=== Z-Score Examples ===")
    # 贵州茅台 (白酒 C15): ROE=30, margin=50, debt=20, growth=15
    moutai = {"roe_latest": 30, "net_margin": 50, "debt_ratio": 20, "revenue_cagr_3y": 15, "industry": "C15"}
    result = score_with_industry_benchmarks(moutai)
    print(f"  茅台 (C15): z_roe={result['z_roe']} z_margin={result['z_margin']} "
          f"z_debt={result['z_debt']} z_growth={result['z_growth']} "
          f"quality={result['industry_quality_score']}")

    # 工商银行 (银行 J66): ROE=10, margin=35, debt=78, growth=5
    icbc = {"roe_latest": 10, "net_margin": 35, "debt_ratio": 78, "revenue_cagr_3y": 5, "industry": "J66"}
    result = score_with_industry_benchmarks(icbc)
    print(f"  工商银行 (J66): z_roe={result['z_roe']} z_margin={result['z_margin']} "
          f"z_debt={result['z_debt']} z_growth={result['z_growth']} "
          f"quality={result['industry_quality_score']}")

    # 中芯国际 (半导体 C39): ROE=5, margin=10, debt=35, growth=20
    smic = {"roe_latest": 5, "net_margin": 10, "debt_ratio": 35, "revenue_cagr_3y": 20, "industry": "C39"}
    result = score_with_industry_benchmarks(smic)
    print(f"  中芯国际 (C39): z_roe={result['z_roe']} z_margin={result['z_margin']} "
          f"z_debt={result['z_debt']} z_growth={result['z_growth']} "
          f"quality={result['industry_quality_score']}")
