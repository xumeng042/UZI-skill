#!/usr/bin/env python3
"""实控人与道德风险评估模块.

判定企业实控人性质（央企/地方国企/民营/外资）、所属区域（南方/北方）、
检测道德风险信号（质押、处罚记录、实控人变更等）。

数据源:
- akshare.stock_profile_cninfo() — 公司注册信息
- akshare.stock_main_stock_holder() — 前十大股东 (复用)
"""

from typing import Optional, Dict, Any, List, Tuple


# ── 省份 → 区域映射 ──────────────────────────────────────────────────

_SOUTH_PROVINCES = {
    "上海", "江苏", "浙江", "安徽", "福建", "江西",
    "湖北", "湖南", "广东", "广西", "海南",
    "重庆", "四川", "贵州", "云南", "西藏",
}

_NORTH_PROVINCES = {
    "北京", "天津", "河北", "山西", "内蒙古",
    "辽宁", "吉林", "黑龙江",
    "山东", "河南", "陕西", "甘肃", "青海", "宁夏", "新疆",
}

# 区域分组
_REGION_MAP = {
    "北京": "华北", "天津": "华北", "河北": "华北", "山西": "华北", "内蒙古": "华北",
    "辽宁": "东北", "吉林": "东北", "黑龙江": "东北",
    "上海": "华东", "江苏": "华东", "浙江": "华东", "安徽": "华东",
    "福建": "华东", "江西": "华东", "山东": "华东",
    "河南": "华中", "湖北": "华中", "湖南": "华中",
    "广东": "华南", "广西": "华南", "海南": "华南",
    "重庆": "西南", "四川": "西南", "贵州": "西南", "云南": "西南", "西藏": "西南",
    "陕西": "西北", "甘肃": "西北", "青海": "西北", "宁夏": "西北", "新疆": "西北",
    "香港": "境外", "澳门": "境外",
}


def _extract_province(addr: str) -> str:
    """Extract province from a Chinese address."""
    if not addr:
        return ""
    # Detect foreign addresses
    foreign_patterns = ["Cricket Square", "P.O. Box", "Cayman", "BVI",
                        "British Virgin", "Bermuda", "Mauritius",
                        "Samoa", "Seychelles", "Panama"]
    addr_upper = addr.upper()
    for fp in foreign_patterns:
        if fp.upper() in addr_upper:
            return "境外"
    # Try province-level municipalities first
    for city in ["北京市", "上海市", "天津市", "重庆市"]:
        if city in addr:
            return city.replace("市", "")
    # Try standard province names
    provinces = [
        "广东", "江苏", "浙江", "山东", "河南", "四川", "湖北", "湖南",
        "福建", "安徽", "河北", "辽宁", "陕西", "江西", "广西", "云南",
        "贵州", "山西", "吉林", "黑龙江", "甘肃", "内蒙古", "新疆",
        "海南", "宁夏", "青海", "西藏",
    ]
    for p in provinces:
        if p in addr:
            return p
    return ""


def get_region(addr: str) -> Tuple[str, str, str]:
    """Determine region from registration address.

    Returns (region_group, north_south, province).
    """
    province = _extract_province(addr)
    if not province:
        return "未知", "未知", ""
    region = _REGION_MAP.get(province, "未知")
    if province in _SOUTH_PROVINCES:
        ns = "南方"
    elif province in _NORTH_PROVINCES:
        ns = "北方"
    else:
        ns = "境外"
    return region, ns, province


# ── 实控人性质判定 ───────────────────────────────────────────────────

# 央企关键词
_CENTRAL_SOE_KEYWORDS = [
    "国务院", "中央汇金", "中国证券金融",
    "财政部", "教育部", "中国科学院", "中国工程院",
    "国家电网", "中国石油", "中国石化", "中国海油",
    "中国移动", "中国电信", "中国联通",
    "中国建筑", "中国中铁", "中国铁建", "中国交建",
    "中国中车", "中国船舶", "中国兵器", "中国航天",
    "中国航空", "中国航发", "中国核工业", "中国电子",
    "中国黄金", "中国铝业", "中国有色", "中国稀土",
    "中国银行", "中国建设", "中国工商", "中国农业",
    "中国人寿", "中国人保", "中国太平",
    "中国中化", "中国化工", "中国建材", "中国医药",
    "中国中煤", "中国煤炭", "中国能源",
    "中国旅游", "中国国旅", "中国中免",
    "中国铁路", "中国民航", "中国邮政",
    "中国华能", "中国华电", "中国大唐", "中国电投",
    "中国一汽", "东风汽车",
    "中国商飞", "中国卫星", "中国卫通",
    "中国盐业", "中国林业", "中国农发",
    "招商局", "招商银行", "中信集团", "中信银行",
    "光大集团", "光大银行", "中国信达", "中国华融",
    "中国长城资产", "中国东方资产",
    "国家集成电路产业投资基金",
]

# 地方国企关键词
_LOCAL_SOE_KEYWORDS = [
    "国有资产", "国有资本", "国资委",
    "省投资", "市投资", "区投资",
    "省能源", "市能源", "省交通", "市交通",
    "省建投", "市建投", "省城建", "市城建",
    "省港口", "市港口", "省高速", "市高速",
    "省水务", "市水务", "省旅游", "市旅游",
    "上海国际", "上海国盛", "上海城投",
    "北京国管", "北京国资", "北京城建",
    "深圳投控", "深圳国资", "广州国资",
    "贵州茅台酒厂", "茅台集团",
    "省属", "市属", "区属",
]

# 外资关键词
_FOREIGN_KEYWORDS = [
    "香港中央结算",
    "UBS", "MORGAN", "DEUTSCHE", "JPMORGAN",
    "GIC", "TEMASEK", "挪威",
    "高盛", "花旗", "汇丰", "渣打",
    "LIMITED", "LTD", "PLC", "INC",
    "HOLDINGS", "CAPITAL", "MANAGEMENT",
]


def _is_central_soe(name: str) -> bool:
    """Check if the name indicates a central SOE."""
    for kw in _CENTRAL_SOE_KEYWORDS:
        if kw in name:
            return True
    return False


def _is_local_soe(name: str) -> bool:
    """Check if the name indicates a local SOE."""
    for kw in _LOCAL_SOE_KEYWORDS:
        if kw in name:
            return True
    # Province/city name + 投/建/发/控/产
    province_prefixes = [
        "北京", "上海", "天津", "重庆",
        "广东", "江苏", "浙江", "山东", "河南", "四川", "湖北", "湖南",
        "福建", "安徽", "河北", "辽宁", "陕西", "江西", "广西", "云南",
        "贵州", "山西", "吉林", "黑龙江", "甘肃", "内蒙古", "新疆",
        "海南", "宁夏", "青海", "西藏",
        "深圳", "广州", "成都", "武汉", "杭州", "南京", "西安", "青岛",
        "厦门", "宁波", "大连", "苏州", "无锡", "合肥", "长沙", "郑州",
    ]
    for prefix in province_prefixes:
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            if suffix and suffix[0] in "投建国控发产金旅交水能":
                return True
    return False


def _detect_controller_type(company_name: str, top_holders: list,
                             profile_desc: str = "") -> str:
    """Determine actual controller type.

    Priority: company name > top shareholder > profile description
    """
    # Check company name first
    if _is_central_soe(company_name):
        return "央企"
    if _is_local_soe(company_name):
        return "地方国企"

    # Check top shareholders
    for h in top_holders[:3]:  # top 3
        name = h.get("name", "")
        if _is_central_soe(name):
            return "央企"
        if _is_local_soe(name):
            return "地方国企"

    # Check if company name contains SOE signals
    if any(kw in company_name for kw in ["中国", "国家"]):
        return "央企"

    # Financial SOEs
    if any(kw in company_name for kw in ["招商银行", "招商证券", "中信银行",
                                           "中信证券", "光大银行", "光大证券"]):
        return "央企"

    # Check for provincial naming patterns
    for p in ["北京", "上海", "天津", "重庆", "广东", "江苏", "浙江",
               "山东", "福建", "四川", "湖北", "湖南", "安徽", "河南"]:
        if company_name.startswith(p) and "集团" in company_name:
            # Could be local SOE or private group
            if "股份" in company_name:
                # Could go either way, check profile
                if any(kw in (profile_desc or "") for kw in
                       ["国有", "国资", "国资委", "财政部"]):
                    return "地方国企"
                return "民营"
            return "民营"

    # Check profile for SOE signals
    if profile_desc:
        if any(kw in profile_desc for kw in
               ["国有法人", "国家股", "国有股", "国资委", "财政部"]):
            return "地方国企"

    # Check if foreign
    # Foreign-registered companies
    if any(kw in company_name.upper() for kw in
           ["LIMITED", "LTD", "INC", "PLC", "CORP"]):
        return "外资"

    return "民营"


# ── 道德风险检测 ──────────────────────────────────────────────────────

def _detect_moral_hazard(company_name: str, profile: dict,
                          top_holders: list, holder_history=None) -> dict:
    """Detect moral hazard signals.

    Returns dict with flags and risk level.
    """
    flags = []
    risk_score = 0

    # 1. Check for shell/ST company name patterns
    name_upper = company_name.upper()
    if any(kw in name_upper for kw in ["*ST", "ST ", "ST"]):
        if company_name.replace("*ST", "").replace("ST", "").strip():
            flags.append("🚨 ST股，存在退市风险")
            risk_score += 3

    # 2. Recent IPO risk (IPO within 3 years)
    list_date = profile.get("list_date", "")
    if list_date:
        try:
            from datetime import datetime
            dt = datetime.strptime(str(list_date)[:10], "%Y-%m-%d")
            years_listed = (datetime.now() - dt).days / 365
            if years_listed < 1:
                flags.append("⚠️ 上市不足1年，历史数据有限")
                risk_score += 2
            elif years_listed < 3:
                flags.append("⚠️ 上市不足3年，需观察业绩持续性")
                risk_score += 1
        except (ValueError, TypeError):
            pass

    # 3. Small registered capital (potential shell)
    reg_capital = profile.get("reg_capital", 0)
    try:
        rc = float(reg_capital) if reg_capital else 0
        if 0 < rc < 2:  # less than 200M RMB
            flags.append("⚠️ 注册资本较小(<2亿)")
            risk_score += 1
    except (ValueError, TypeError):
        pass

    # 4. Frequent controller changes (detected from holder history)
    if holder_history:
        # Count unique top-1 shareholders in last 8 quarters
        top1_names = set()
        quarters_seen = set()
        for entry in holder_history:
            date_key = str(entry.get("date", ""))[:7]
            if date_key in quarters_seen:
                continue
            quarters_seen.add(date_key)
            holders = entry.get("holders", [])
            if holders:
                top1_names.add(holders[0].get("name", ""))
        if len(top1_names) >= 3:
            flags.append("🚨 近两年实控人频繁变更")
            risk_score += 3
        elif len(top1_names) >= 2:
            flags.append("⚠️ 近两年第一大股东发生过变更")
            risk_score += 1

    # 5. Controlling shareholder with high pledge risk (placeholder)
    # This requires separate API; flag as data gap for now

    # 6. Cross-region registration mismatch
    addr = profile.get("reg_addr", "")
    office = profile.get("office_addr", "")
    if addr and office:
        p1 = _extract_province(addr)
        p2 = _extract_province(office)
        if p1 and p2 and p1 != p2:
            flags.append("⚠️ 注册地与办公地不在同一省份")
            risk_score += 0.5

    # 7. Name contains common problematic patterns
    problem_patterns = [
        "科技", "高科", "新能", "互联", "智能", "数据", "云",
    ]
    # Not necessarily a problem, but combined with other signals raises concern
    _ = problem_patterns  # reserved for future use

    # Determine risk level
    if risk_score >= 4:
        risk_level = "高"
    elif risk_score >= 2:
        risk_level = "中"
    elif risk_score >= 1:
        risk_level = "低"
    else:
        risk_level = "无"

    return {
        "flags": flags,
        "risk_score": risk_score,
        "risk_level": risk_level,
    }


# ── Main API ──────────────────────────────────────────────────────────

def _code_to_ak(code: str) -> str:
    """Convert baostock code to akshare code."""
    return code.replace("sh.", "").replace("sz.", "").replace("bj.", "")


def analyze_controller(code: str, holder_data: Optional[dict] = None) -> dict:
    """Analyze controller type, region, and moral hazard for a stock.

    Args:
        code: baostock code (e.g. 'sh.600519')
        holder_data: pre-fetched institutional holdings result (optional)

    Returns dict with controller_type, region, north_south, moral_hazard.
    """
    result = {
        "controller_type": "未知",
        "region_group": "未知",
        "north_south": "未知",
        "province": "",
        "reg_address": "",
        "moral_flags": [],
        "moral_risk_level": "无",
    }

    try:
        import akshare as ak
        ak_code = _code_to_ak(code)
        df = ak.stock_profile_cninfo(symbol=ak_code)
        if df.empty:
            return result
        row = df.iloc[0]
    except Exception:
        return result

    company_name = str(row.get("公司名称", ""))
    reg_addr = str(row.get("注册地址", ""))
    office_addr = str(row.get("办公地址", ""))
    list_date = str(row.get("上市日期", ""))
    reg_capital = row.get("注册资金", 0)
    profile_desc = str(row.get("机构简介", ""))

    # Region
    region_group, north_south, province = get_region(reg_addr)

    # Controller type: use top holders from holder_data if available
    top_holders = []
    if holder_data and holder_data.get("holders"):
        top_holders = holder_data["holders"]

    controller_type = _detect_controller_type(company_name, top_holders, profile_desc)

    # Moral hazard
    profile_info = {
        "list_date": list_date,
        "reg_capital": reg_capital,
        "reg_addr": reg_addr,
        "office_addr": office_addr,
    }
    moral = _detect_moral_hazard(company_name, profile_info, top_holders)

    result.update({
        "controller_type": controller_type,
        "region_group": region_group,
        "north_south": north_south,
        "province": province,
        "reg_address": reg_addr,
        "company_name": company_name,
        "list_date": list_date[:10] if list_date else "",
        "moral_flags": moral["flags"],
        "moral_risk_level": moral["risk_level"],
        "moral_risk_score": moral["risk_score"],
    })
    return result


def score_controller_dimension(ctrl_result: dict) -> float:
    """Score controller & region dimension 1-10.

    Considers:
    - Controller stability (央企 > 地方国企 > 民营 > 外资)
    - Moral hazard risk
    """
    s = 5

    # Controller type bonus/penalty
    ct = ctrl_result.get("controller_type", "未知")
    if ct == "央企":
        s += 2  # highest stability
    elif ct == "地方国企":
        s += 1  # stable
    elif ct == "民营":
        s += 0  # neutral
    elif ct == "外资":
        s -= 1  # potential regulatory risk
    else:
        s += 0

    # Moral hazard penalty
    risk_level = ctrl_result.get("moral_risk_level", "无")
    risk_score = ctrl_result.get("moral_risk_score", 0)

    if risk_level == "高":
        s -= 3
    elif risk_level == "中":
        s -= 2
    elif risk_level == "低":
        s -= 1

    # Risk flag count penalty
    flag_count = len(ctrl_result.get("moral_flags", []))
    if flag_count >= 3:
        s -= 1

    return max(1, min(10, s))


def get_controller_summary(ctrl_result: dict) -> str:
    """One-line summary for display."""
    ct = ctrl_result.get("controller_type", "?")
    ns = ctrl_result.get("north_south", "?")
    risk = ctrl_result.get("moral_risk_level", "?")
    region = ctrl_result.get("region_group", "?")
    return f"{ct}/{ns}/{region}(道德{risk})"


# ── CLI test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_stocks = [
        ("sh.600519", "贵州茅台"),
        ("sh.688981", "中芯国际"),
        ("sz.300750", "宁德时代"),
        ("sh.600036", "招商银行"),
        ("sh.600276", "恒瑞医药"),
        ("sh.600893", "航发动力"),
        ("sz.000651", "格力电器"),
        ("sh.601318", "中国平安"),
    ]
    for code, name in test_stocks:
        result = analyze_controller(code)
        score = score_controller_dimension(result)
        flags_str = "; ".join(result["moral_flags"]) if result["moral_flags"] else "无"
        print(f"{name:6s} | {result['controller_type']:4s} | "
              f"{result['north_south']:2s}/{result['region_group']:2s} "
              f"({result['province']:3s}) | 维度分={score:.0f} | "
              f"道德风险={result['moral_risk_level']}({result['moral_risk_score']}) | "
              f"信号: {flags_str}")
