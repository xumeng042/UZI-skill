#!/usr/bin/env python3
"""行业周期与季节评估模块.

基于现有财务数据识别个股所处行业周期阶段、检测下行风险信号、
评估季节性特征。不使用额外 API 调用，完全基于 fetch_one_stock()
已获取的数据。

核心逻辑:
- 周期阶段: 通过收入/利润增速轨迹、ROE 趋势、毛利率稳定性判断
- 下行风险: 营收持续下滑、毛利压缩、ROE 恶化、负债攀升
- 季节性: 基于行业特征和季度收入分布规律
"""

from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime


# ── 行业季节特征 ──────────────────────────────────────────────────────
# 基于 A 股历史规律的行业季节性倾向（旺季月份）

INDUSTRY_SEASONALITY = {
    # 消费类 — Q4+Q1 旺季（春节+年终消费）
    "C15": (10, 3),   # 酒、饮料、精制茶 → 10月-3月旺季
    "C14": (10, 3),   # 食品制造 → 同上
    "C13": (9, 2),    # 农副食品加工
    "F52": (10, 3),   # 零售业 → 年底消费旺季
    "F51": (10, 3),   # 批发业

    # 周期类 — Q2-Q3 旺季（施工季节）
    "C30": (4, 9),    # 非金属矿物 → 建筑旺季
    "C26": (4, 9),    # 化学原料 → 工业旺季
    "C32": (4, 9),    # 有色金属
    "C35": (4, 9),    # 专用设备
    "C34": (4, 9),    # 通用设备

    # 金融类 — Q1+Q4 旺季（信贷投放+年底结算）
    "J66": (1, 4),    # 银行 → 信贷开门红
    "J67": (1, 4),    # 券商 → 春季行情

    # 电力/能源 — Q1+Q4 旺季（供暖季）
    "D44": (10, 3),   # 电力、热力
    "D45": (10, 3),   # 燃气

    # 汽车 — Q4+Q1 旺季（年底冲量+春节前购车）
    "C36": (10, 3),

    # 医药 — 全年平稳，Q1 略强（年报季+流感季）
    "C27": (1, 4),
    "Q84": (1, 4),

    # 科技 — Q2+Q4 偏强（新品发布+年底采购）
    "C39": (4, 12),   # 计算机通信电子
    "I65": (4, 12),   # 软件

    # 农业 — Q2-Q3 旺季（种植/收获）
    "A01": (4, 9),

    # 房地产/建筑 — Q2-Q3 旺季
    "E47": (4, 9),    # 房屋建筑业
    "E48": (4, 9),    # 土木工程

    # 交通运输 — Q3-Q4 旺季（出行+货运高峰）
    "G54": (7, 12),   # 道路运输
    "G56": (7, 12),   # 航空运输
}


def get_seasonal_phase(industry_code: str) -> Tuple[str, str]:
    """Get current seasonal phase for an industry.

    Returns (phase, note) where phase is '旺季'/'淡季'/'平稳'.
    """
    if not industry_code:
        return "平稳", ""

    prefix = industry_code[:3] if len(industry_code) >= 3 else industry_code[:2]
    seasonal = INDUSTRY_SEASONALITY.get(prefix)

    if seasonal is None:
        return "平稳", "无明显季节性"

    now = datetime.now()
    month = now.month
    peak_start, peak_end = seasonal

    if peak_start <= peak_end:
        in_season = peak_start <= month <= peak_end
    else:  # wraps around year (e.g., 10月 to 3月)
        in_season = month >= peak_start or month <= peak_end

    if in_season:
        return "旺季", f"{peak_start}月-{peak_end}月为行业旺季"
    else:
        # Near peak (within 1 month)
        near_peak = False
        for boundary in [peak_start, peak_end]:
            diff = abs(month - boundary)
            if diff <= 1 or diff >= 11:  # wrap around for year boundary
                near_peak = True
                break
        if near_peak:
            return "旺季临近", f"即将进入{peak_start}月-{peak_end}月旺季"
        return "淡季", f"旺季为{peak_start}月-{peak_end}月"


# ── 周期阶段评估 ──────────────────────────────────────────────────────

def assess_cycle(m: dict) -> dict:
    """Assess industry/business cycle stage from financial metrics.

    Analyzes revenue trajectory, ROE trend, margin stability, and
    profit consistency to determine whether a company is in:
    - 成长期 (Growth): accelerating revenue, expanding margins
    - 成熟期 (Mature): stable growth, consistent margins
    - 高位盘整 (Plateau): growth decelerating, margins stable
    - 下行期 (Decline): declining revenue, compressing margins

    Returns dict with cycle_stage, cycle_score, risk_flags, signals.
    """
    signals = {}
    risk_flags = []
    cycle_score = 0

    # ── 1. Revenue trajectory ──────────────────────────────────────
    rev_cagr = m.get("revenue_cagr_3y", 0) or 0
    rev_growth = m.get("revenue_growth", 0) or 0
    rev_up = m.get("revenue_up_years", 0) or 0

    # Acceleration/deceleration
    growth_delta = rev_growth - rev_cagr

    if rev_cagr > 20 and growth_delta > 5:
        signals["revenue"] = "高增长加速"
        cycle_score += 3
    elif rev_cagr > 20 and growth_delta >= -5:
        signals["revenue"] = "高增长稳健"
        cycle_score += 2
    elif rev_cagr > 10 and growth_delta > 3:
        signals["revenue"] = "中速增长加速"
        cycle_score += 2
    elif rev_cagr > 10:
        signals["revenue"] = "中速增长"
        cycle_score += 1
    elif rev_cagr > 5:
        signals["revenue"] = "低速增长"
        cycle_score += 0
    elif rev_cagr > 0 and growth_delta < -5:
        signals["revenue"] = "增长显著放缓"
        cycle_score -= 1
        risk_flags.append("⚠️ 收入增速显著放缓")
    elif rev_cagr > 0:
        signals["revenue"] = "缓慢增长"
        cycle_score += 0
    elif rev_cagr <= 0 and rev_growth > 0:
        signals["revenue"] = "触底反弹"
        cycle_score += 1
    else:
        signals["revenue"] = "持续下滑"
        cycle_score -= 2
        risk_flags.append("🚨 收入持续下滑")

    # Revenue decline for 2+ years
    if rev_up <= 1:
        risk_flags.append("⚠️ 近3年收入增长不足")

    # ── 2. Profit trajectory ─────────────────────────────────────
    profit_cagr = m.get("profit_cagr_3y", 0) or 0
    profit_up = m.get("profit_up_years", 0) or 0
    net_margin = m.get("net_margin", 0) or 0

    if profit_cagr > 25:
        signals["profit"] = "利润高增长"
        cycle_score += 2
    elif profit_cagr > 15:
        signals["profit"] = "利润稳健增长"
        cycle_score += 1
    elif profit_cagr > 5:
        signals["profit"] = "利润低速增长"
        cycle_score += 0
    elif profit_cagr > 0:
        signals["profit"] = "利润微增"
        cycle_score -= 1
    elif profit_cagr <= 0 and profit_up >= 2:
        signals["profit"] = "利润见底"
        cycle_score += 0
    else:
        signals["profit"] = "利润下滑"
        cycle_score -= 2
        risk_flags.append("🚨 利润持续下滑")

    # Net margin check
    if net_margin < 5:
        risk_flags.append("⚠️ 净利率极低(<5%)")
    elif net_margin < 0:
        risk_flags.append("🚨 净利润为负")

    # ── 3. ROE trend ────────────────────────────────────────────
    roe_now = m.get("roe_latest", 0) or 0
    roe_min = m.get("roe_5y_min", 0) or 0
    roe_min_15 = m.get("roe_min_15", False)
    roe_min_10 = m.get("roe_min_10", False)

    roe_stability = roe_now - roe_min

    if roe_now >= 20 and roe_min_15:
        signals["roe"] = "持续高ROE"
        cycle_score += 2
    elif roe_now >= 15 and roe_min_10:
        signals["roe"] = "稳定ROE"
        cycle_score += 1
    elif roe_now >= 10:
        signals["roe"] = "中等ROE"
        cycle_score += 0
    elif roe_now >= 5:
        signals["roe"] = "ROE偏低"
        cycle_score -= 1
    else:
        signals["roe"] = "ROE过低"
        cycle_score -= 2
        risk_flags.append("🚨 ROE 长期低于5%")

    # ROE declining trend
    if roe_stability > 15 and roe_now < roe_min + 5:
        signals["roe"] += "(高位回落)"
        cycle_score -= 1
        risk_flags.append("⚠️ ROE 从高位显著回落")

    # ── 4. Gross margin stability ──────────────────────────────
    gm = m.get("gross_margin", 0) or 0
    gm_stable = m.get("gross_margin_stable", False)

    if gm >= 40 and gm_stable:
        signals["margin"] = "高毛利稳定"
        cycle_score += 1
    elif gm >= 40:
        signals["margin"] = "高毛利波动"
        risk_flags.append("⚠️ 毛利率波动较大")
    elif gm >= 20 and gm_stable:
        signals["margin"] = "中等毛利稳定"
        cycle_score += 0
    elif gm >= 20:
        signals["margin"] = "中等毛利波动"
    elif gm < 20 and not gm_stable:
        signals["margin"] = "低毛利压缩"
        cycle_score -= 1
        risk_flags.append("⚠️ 毛利率偏低且波动")
    else:
        signals["margin"] = "低毛利"

    # ── 5. Debt risk ───────────────────────────────────────────
    debt = m.get("debt_ratio", 50) or 50
    if debt > 70:
        risk_flags.append("🚨 负债率超过70%")
        cycle_score -= 1
    elif debt > 60:
        risk_flags.append("⚠️ 负债率偏高(>60%)")
    elif debt < 20:
        signals["debt"] = "低杠杆"

    # ── 6. Profit quality ──────────────────────────────────────
    consec = m.get("consecutive_profit_years", 0) or 0
    if consec >= 5:
        signals["profit_quality"] = f"连续{consec}年盈利"
        cycle_score += 1
    elif consec <= 1:
        risk_flags.append("⚠️ 盈利持续性不足")

    # ── 7. YoY growth signals ─────────────────────────────────
    yoy_ni = m.get("yoy_ni", 0) or 0
    if yoy_ni < -20:
        risk_flags.append(f"🚨 净利润同比大跌{yoy_ni:.0f}%")
        cycle_score -= 1
    elif yoy_ni < -10:
        risk_flags.append(f"⚠️ 净利润同比下降{yoy_ni:.0f}%")

    # ── 8. Determine cycle stage ───────────────────────────────
    if cycle_score >= 4:
        cycle_stage = "成长期"
        outlook = "正面"
    elif cycle_score >= 1:
        cycle_stage = "成熟期"
        outlook = "正面"
    elif cycle_score >= -1:
        cycle_stage = "高位盘整"
        outlook = "中性"
    elif cycle_score >= -3:
        cycle_stage = "下行初期"
        outlook = "谨慎"
    else:
        cycle_stage = "下行期"
        outlook = "负面"

    risk_level = "低"
    if len([f for f in risk_flags if f.startswith("🚨")]) >= 2:
        risk_level = "高"
    elif len([f for f in risk_flags if f.startswith("🚨")]) >= 1:
        risk_level = "中高"
    elif len(risk_flags) >= 2:
        risk_level = "中"
    elif len(risk_flags) >= 1:
        risk_level = "低"

    return {
        "cycle_stage": cycle_stage,
        "cycle_score_raw": cycle_score,
        "outlook": outlook,
        "signals": signals,
        "risk_flags": risk_flags,
        "risk_level": risk_level,
        "risk_count": len(risk_flags),
    }


def score_cycle_dimension(cycle_result: dict) -> float:
    """Convert cycle assessment to a 1-10 score.

    Maps cycle_stage to base score, then adjusts for risk flags.
    """
    stage = cycle_result["cycle_stage"]
    base = {
        "成长期": 9,
        "成熟期": 7,
        "高位盘整": 5,
        "下行初期": 3,
        "下行期": 1,
    }.get(stage, 5)

    # Adjust for risk flags
    risk_count = cycle_result["risk_count"]
    if risk_count >= 3:
        base -= 2
    elif risk_count >= 2:
        base -= 1
    elif risk_count >= 1:
        base -= 0.5

    # Boost for low-risk with strong signals
    signals = cycle_result["signals"]
    positive_signals = sum(1 for v in signals.values()
                          if any(kw in str(v) for kw in
                                ["高增长", "加速", "稳定", "持续高ROE", "低杠杆"]))
    if positive_signals >= 3:
        base += 1

    return max(1, min(10, base))


def get_cycle_summary(cycle_result: dict) -> str:
    """One-line summary for display."""
    stage = cycle_result["cycle_stage"]
    risk = cycle_result["risk_level"]
    return f"{stage}(风险{risk})"


# ── CLI test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from screen_stocks import fetch_one_stock
    import baostock as bs

    bs.login()
    try:
        test_stocks = [
            ("sh.600519", "贵州茅台"),
            ("sh.688981", "中芯国际"),
            ("sz.300750", "宁德时代"),
            ("sh.600036", "招商银行"),
            ("sh.600276", "恒瑞医药"),
            ("sh.600893", "航发动力"),
            ("sz.000651", "格力电器"),
        ]
        for code, name in test_stocks:
            m = fetch_one_stock(code, name)
            if m is None:
                print(f"{name}: fetch failed")
                continue

            cycle = assess_cycle(m)
            score = score_cycle_dimension(cycle)
            ind = m.get("industry", "")
            phase, note = get_seasonal_phase(ind)

            flags_str = "; ".join(cycle["risk_flags"]) if cycle["risk_flags"] else "无"
            print(f"{name:6s} | 阶段={cycle['cycle_stage']:5s} | 维度分={score:.1f} | "
                  f"展望={cycle['outlook']:2s} | 风险={cycle['risk_level']:3s}({cycle['risk_count']}项) | "
                  f"季节={phase} | "
                  f"增速={cycle['signals'].get('revenue','?')} | "
                  f"利润={cycle['signals'].get('profit','?')} | "
                  f"ROE={cycle['signals'].get('roe','?')}")
            if cycle["risk_flags"]:
                print(f"       ⚡ 风险: {flags_str}")
    finally:
        bs.logout()
