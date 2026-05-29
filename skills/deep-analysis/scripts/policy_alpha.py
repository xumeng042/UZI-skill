#!/usr/bin/env python3
"""十五五规划 (2026-2030) 政策面增强模块.

通过行业映射量化政策风口红利，结合 YoY 增长数据，为选股提供政策 α 维度。

数据源:
- baostock.query_stock_industry() — 行业分类
- baostock.query_growth_data() — 同比增长率
"""

import sys
import os
from typing import Optional

# ── 十五五规划重点领域 → 证监会行业映射 ─────────────────────────────

# 高优先级 (政策分 +15): 国家战略核心方向
POLICY_TIER_1 = {
    "半导体/芯片": [
        "C39计算机、通信和其他电子设备制造业",  # 芯片设计/制造/封装
    ],
    "人工智能/AI": [
        "I65软件和信息技术服务业",  # AI软件/平台
        "I64互联网和相关服务",     # 互联网AI应用
    ],
    "新能源/碳中和": [
        "D44电力、热力生产和供应业",      # 新能源发电(光伏/风电/核电)
        "C38电气机械和器材制造业",        # 光伏设备/储能/电网设备
        "N77生态保护和环境治理业",        # 碳捕集/环保
    ],
    "商业航天/低空经济": [
        "C37铁路、船舶、航空航天和其他运输设备制造业",  # 航天/大飞机/低空
    ],
    "生物医药/创新药": [
        "C27医药制造业",           # 创新药/生物制品/中药
        "M73研究和试验发展",       # CRO/CDMO
    ],
    "高端制造/新质生产力": [
        "C35专用设备制造业",       # 半导体设备/光伏设备/锂电设备
        "C34通用设备制造业",       # 机器人/工业母机
        "C40仪器仪表制造业",       # 精密仪器/传感器
    ],
}

# 中优先级 (政策分 +10): 国家重点支持
POLICY_TIER_2 = {
    "新能源车/智能驾驶": [
        "C36汽车制造业",           # 整车/零部件/智驾
    ],
    "新材料/战略资源": [
        "C30非金属矿物制品业",     # 碳纤维/特种陶瓷
        "C26化学原料和化学制品制造业",  # 化工新材料/电子化学品
        "C32有色金属冶炼和压延加工业",  # 稀土/锂/钴
    ],
    "粮食安全/种业": [
        "A01农业",                 # 种业/种植
        "C13农副食品加工业",       # 农产品加工
    ],
    "数据要素/信创": [
        "I63电信、广播电视和卫星传输服务",  # 5G/6G/卫星互联网
    ],
    "量子科技": [
        "C39计算机、通信和其他电子设备制造业",  # 量子计算/量子通信
    ],
    "国防军工": [
        "C37铁路、船舶、航空航天和其他运输设备制造业",  # 军工
    ],
}

# 政策红利 (政策分 +5): 受益于政策导向
POLICY_TIER_3 = {
    "大消费/内循环": [
        "C15酒、饮料和精制茶制造业",
        "F51批发业",
        "F52零售业",
    ],
    "养老/银发经济": [
        "Q84卫生",                 # 医疗服务
    ],
    "数字经济/金融科技": [
        "J66货币金融服务",         # 银行/数字人民币
        "J67资本市场服务",         # 券商/金融IT
    ],
    "食品/消费升级": [
        "C14食品制造业",           # 食品饮料
    ],
    "电力改革": [
        "D45燃气生产和供应业",     # 天然气
    ],
}

# 构建快速查找表: industry_code → (policy_name, score)
_LOOKUP: dict = {}
for _name, _codes in POLICY_TIER_1.items():
    for _c in _codes:
        _LOOKUP[_c] = (_name, 15)
for _name, _codes in POLICY_TIER_2.items():
    for _c in _codes:
        if _c not in _LOOKUP:
            _LOOKUP[_c] = (_name, 10)
for _name, _codes in POLICY_TIER_3.items():
    for _c in _codes:
        if _c not in _LOOKUP:
            _LOOKUP[_c] = (_name, 5)


# ── Public API ───────────────────────────────────────────────────────────

def get_policy_score(industry: str) -> tuple:
    """Return (policy_name, score) for a CSRC industry code. (None, 0) if no match."""
    if not industry:
        return None, 0
    # Try exact match first
    if industry in _LOOKUP:
        return _LOOKUP[industry]
    # Try partial match (industry code prefix)
    for code, (name, score) in _LOOKUP.items():
        if industry.startswith(code[:2]):
            return name, score
    return None, 0


def get_policy_tier_name(score: int) -> str:
    if score >= 15:
        return "核心战略"
    if score >= 10:
        return "重点支持"
    if score >= 5:
        return "政策红利"
    return "—"


def fetch_industry(bs, code: str) -> Optional[str]:
    """Get industry classification for one stock. Must be called within baostock session."""
    try:
        rs = bs.query_stock_industry(code=code)
        if rs.error_code != "0":
            return None
        df = rs.get_data()
        if df.empty:
            return None
        # Return the first (most recent) industry entry
        row = df.iloc[0]
        industry = str(row.get("industry", "")).strip()
        return industry if industry else None
    except Exception:
        return None


def fetch_growth_data(bs, code: str, year: int = 2024, quarter: int = 4) -> dict:
    """Get YoY growth rates. Returns {yoy_equity, yoy_asset, yoy_ni, yoy_eps}."""
    try:
        rs = bs.query_growth_data(code=code, year=year, quarter=quarter)
        if rs.error_code != "0":
            return {}
        df = rs.get_data()
        if df.empty:
            return {}
        row = df.iloc[-1]
        return {
            "yoy_equity": _f_g(row, "YOYEquity"),
            "yoy_asset": _f_g(row, "YOYAsset"),
            "yoy_ni": _f_g(row, "YOYNI"),
            "yoy_eps": _f_g(row, "YOYEPSBasic"),
        }
    except Exception:
        return {}


def _f_g(row, key, default=0):
    try:
        v = row.get(key)
        return round(float(v) * 100, 1) if v is not None and v != "" else default
    except (ValueError, TypeError):
        return default


def score_policy_dimension(policy_name: str, policy_score: int, growth_data: dict) -> float:
    """Score the policy dimension 1-10.

    Combines:
    - Policy tier alignment (0-7 points based on tier)
    - Growth momentum from YoY data (0-3 points)
    """
    s = 0

    # Policy tier contribution
    if policy_score >= 15:
        s += 7
    elif policy_score >= 10:
        s += 5
    elif policy_score >= 5:
        s += 3
    else:
        s += 1  # baseline, no policy tailwind

    # Growth momentum contribution
    yoy_ni = growth_data.get("yoy_ni", 0)
    yoy_eps = growth_data.get("yoy_eps", 0)
    avg_growth = (yoy_ni + yoy_eps) / 2 if yoy_ni and yoy_eps else max(yoy_ni, yoy_eps)

    if avg_growth > 30:
        s += 3
    elif avg_growth > 20:
        s += 2
    elif avg_growth > 10:
        s += 1
    elif avg_growth < -10:
        s -= 1

    return max(1, min(10, s))


# ── CLI test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import baostock as bs
    bs.login()
    try:
        # Test: show policy mapping for all industries
        print("=== 十五五规划 重点领域 → 行业映射 ===\n")
        seen = set()
        for tier_name, tier_dict, marker in [
            ("核心战略 (L1, +15)", POLICY_TIER_1, "🔴"),
            ("重点支持 (L2, +10)", POLICY_TIER_2, "🟡"),
            ("政策红利 (L3, +5)", POLICY_TIER_3, "🟢"),
        ]:
            print(f"## {marker} {tier_name}")
            for area, codes in tier_dict.items():
                print(f"  {area}:")
                for c in codes:
                    print(f"    - {c}")
            print()

        # Test: sample stock industry lookup
        print("=== 样本股票政策检测 ===\n")
        test_stocks = [
            ("sh.600519", "贵州茅台"),
            ("sh.688981", "中芯国际"),
            ("sz.300750", "宁德时代"),
            ("sh.600276", "恒瑞医药"),
            ("sh.600036", "招商银行"),
            ("sh.601012", "隆基绿能"),
            ("sh.600893", "航发动力"),
            ("sz.002415", "海康威视"),
        ]
        for code, name in test_stocks:
            ind = fetch_industry(bs, code)
            pname, pscore = get_policy_score(ind or "")
            growth = fetch_growth_data(bs, code)
            dim_score = score_policy_dimension(pname, pscore, growth)
            print(f"  {name:6s} ({code}): 行业={ind or '?'} → "
                  f"政策={pname or '无'} 分={pscore} 增长={growth} 维度分={dim_score:.0f}")
    finally:
        bs.logout()
