#!/usr/bin/env python3
"""大资金持仓分析模块.

通过 akshare 获取前十大股东数据，识别中央汇金、社保基金、险资、
北向资金、QFII 等大资金持仓变化，为选股提供机构信心维度。

数据源:
- akshare.stock_main_stock_holder() — 前十大股东历史数据
- akshare.stock_fund_stock_holder() — 基金持仓 (可选)
- akshare.stock_hsgt_individual_em() — 北向资金每日流向 (可选)
"""

from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from collections import defaultdict


# ── 机构分类关键词 ──────────────────────────────────────────────────

# 中央汇金 / 证金 — 国家队
NATIONAL_TEAM = [
    "中央汇金", "中国证券金融", "证金",
]

# 社保基金 — 养老金
SOCIAL_SECURITY = [
    "全国社保基金", "基本养老保险基金", "社保基金",
]

# 险资 — 保险资金
INSURANCE = [
    "人寿保险", "财产保险", "平安保险", "太平洋保险",
    "泰康保险", "新华保险", "人民保险", "阳光保险",
    "生命人寿", "太平人寿", "华夏人寿", "前海人寿",
    "和谐健康", "安邦保险", "大家保险",
]

# 北向资金 — 香港中央结算 (沪股通/深股通汇总)
NORTHBOUND = [
    "香港中央结算",
]

# QFII / 外资
QFII = [
    "UBS AG", "MORGAN STANLEY", "DEUTSCHE BANK",
    "GIC PRIVATE", "挪威中央银行", "TEMASEK",
    "奥本海默", "高盛", "JPMORGAN", "CITIGROUP",
    "MERRILL LYNCH", "CREDIT SUISSE", "BARCLAYS",
    "法国巴黎银行", "BNP PARIBAS", "ABU DHABI",
    "高瓴", "高毅", "淡马锡",
]

# 公募基金关键词 (用于识别但权重较低)
MUTUAL_FUND = [
    "交易型开放式指数", "证券投资基金", "混合型证券投资基金",
    "指数分级证券投资基金",
]


def _contains_any(name: str, keywords: List[str]) -> bool:
    """Check if name contains any keyword (case-insensitive for foreign names)."""
    upper = name.upper()
    for kw in keywords:
        if kw.upper() in upper:
            return True
    return False


def classify_shareholder(name: str) -> Tuple[str, int]:
    """Classify a shareholder name into category.

    Returns (category, weight) where weight indicates significance.
    """
    if not name or name == "nan":
        return "", 0

    if _contains_any(name, NATIONAL_TEAM):
        return "国家队", 5
    if _contains_any(name, SOCIAL_SECURITY):
        return "社保基金", 5
    if _contains_any(name, NORTHBOUND):
        return "北向资金", 4
    if _contains_any(name, INSURANCE):
        return "险资", 4
    if _contains_any(name, QFII):
        return "QFII", 3
    if _contains_any(name, MUTUAL_FUND):
        return "公募基金", 1
    return "其他", 0


def _code_to_ak(code: str) -> str:
    """Convert baostock code ('sh.600519') to akshare code ('600519')."""
    return code.replace("sh.", "").replace("sz.", "").replace("bj.", "")


# ── Main API ──────────────────────────────────────────────────────────

def fetch_institutional(code: str) -> Optional[Dict[str, Any]]:
    """Fetch and analyze institutional holdings for one stock.

    Args:
        code: baostock-format code (e.g. 'sh.600519')

    Returns dict with analysis results, or None on failure.
    """
    ak_code = _code_to_ak(code)
    try:
        import akshare as ak
        df = ak.stock_main_stock_holder(stock=ak_code)
        if df.empty:
            return None
    except Exception:
        return None

    # Get the two most recent quarters
    df["截至日期"] = pd.to_datetime(df["截至日期"])
    dates = sorted(df["截至日期"].unique(), reverse=True)
    if not dates:
        return None

    latest_date = dates[0]
    prev_date = None
    for d in dates:
        if d < latest_date:
            prev_date = d
            break

    # Filter to latest quarter's top 10
    latest_rows = df[df["截至日期"] == latest_date].copy()
    prev_rows = df[df["截至日期"] == prev_date].copy() if prev_date else None

    # Analyze current holders
    holders = []
    institutional_score = 0
    categories_found = set()
    huijin_holds = False
    shebao_holds = False
    xianzi_holds = False
    northbound_holds = False
    qfii_holds = False

    total_institutional_pct = 0.0

    for _, row in latest_rows.iterrows():
        name = str(row.get("股东名称", ""))
        pct = _safe_float(row.get("持股比例", 0))
        shares = _safe_float(row.get("持股数量", 0))

        cat, weight = classify_shareholder(name)

        holders.append({
            "name": name,
            "category": cat,
            "weight": weight,
            "pct": round(pct, 2),
            "shares": shares,
        })

        if cat == "国家队":
            huijin_holds = True
        elif cat == "社保基金":
            shebao_holds = True
        elif cat == "险资":
            xianzi_holds = True
        elif cat == "北向资金":
            northbound_holds = True
        elif cat == "QFII":
            qfii_holds = True

        if weight >= 3:
            categories_found.add(cat)
            total_institutional_pct += pct
            institutional_score += weight

    # Detect quarter-over-quarter changes
    changes = _detect_changes(latest_rows, prev_rows) if prev_date else {}

    # Compute institutional concentration
    num_categories = len(categories_found)

    # Count top funds
    fund_count = sum(1 for h in holders if h["category"] == "公募基金")

    result = {
        "latest_date": latest_date.strftime("%Y-%m-%d"),
        "prev_date": prev_date.strftime("%Y-%m-%d") if prev_date else "",
        "holders": holders,
        "huijin": huijin_holds,
        "shebao": shebao_holds,
        "xianzi": xianzi_holds,
        "northbound": northbound_holds,
        "qfii": qfii_holds,
        "num_categories": num_categories,
        "total_inst_pct": round(total_institutional_pct, 2),
        "fund_count": fund_count,
        "changes": changes,
    }
    return result


def _safe_float(val, default=0.0):
    try:
        return float(val) if val and str(val) != "nan" else default
    except (ValueError, TypeError):
        return default


def _detect_changes(latest_df, prev_df) -> dict:
    """Detect changes in institutional holdings between two quarters."""
    prev_names = set()
    prev_holdings = {}
    for _, row in prev_df.iterrows():
        name = str(row.get("股东名称", ""))
        if name and name != "nan":
            pct = _safe_float(row.get("持股比例", 0))
            prev_names.add(name)
            prev_holdings[name] = pct

    changes = {"new": [], "exited": [], "increased": [], "decreased": []}

    for _, row in latest_df.iterrows():
        name = str(row.get("股东名称", ""))
        if not name or name == "nan":
            continue
        pct = _safe_float(row.get("持股比例", 0))
        cat, _ = classify_shareholder(name)

        if name not in prev_names:
            if cat not in ("", "其他", "公募基金"):
                changes["new"].append({"name": name, "category": cat, "pct": pct})
        elif name in prev_holdings:
            old_pct = prev_holdings[name]
            diff = pct - old_pct
            if diff > 0.1 and cat not in ("", "其他"):
                changes["increased"].append({"name": name, "category": cat, "diff": round(diff, 2)})
            elif diff < -0.1 and cat not in ("", "其他"):
                changes["decreased"].append({"name": name, "category": cat, "diff": round(diff, 2)})

    # Detect exited institutions
    latest_names = set()
    for _, row in latest_df.iterrows():
        name = str(row.get("股东名称", ""))
        if name and name != "nan":
            latest_names.add(name)

    for name in prev_names - latest_names:
        cat, _ = classify_shareholder(name)
        if cat not in ("", "其他", "公募基金"):
            changes["exited"].append({"name": name, "category": cat})

    return changes


# ── Scoring ───────────────────────────────────────────────────────────

def score_institutional(result: Optional[Dict]) -> float:
    """Score institutional dimension 1-10."""
    if result is None:
        return 5  # neutral — no data

    s = 5

    # Core institution presence (0-4)
    if result["huijin"]:
        s += 2
    if result["shebao"]:
        s += 2
    if result["xianzi"]:
        s += 1.5
    if result["northbound"]:
        s += 1.5
    if result["qfii"]:
        s += 1

    # Institutional diversity bonus (0-2)
    nc = result["num_categories"]
    if nc >= 4:
        s += 2
    elif nc >= 3:
        s += 1.5
    elif nc >= 2:
        s += 1
    elif nc >= 1:
        s += 0.5

    # Recent changes (0-2)
    changes = result.get("changes", {})
    new_count = len(changes.get("new", []))
    inc_count = len(changes.get("increased", []))
    dec_count = len(changes.get("decreased", []))
    exit_count = len(changes.get("exited", []))

    if new_count > 0 and inc_count > 0:
        s += 2  # institutions entering AND accumulating
    elif new_count > 0:
        s += 1.5  # new institutions entering
    elif inc_count > 0:
        s += 1  # existing institutions accumulating
    elif dec_count > 0 and exit_count == 0:
        s += 0  # mild reduction, neutral
    elif exit_count > 0:
        s -= 1  # institutions exiting
    if dec_count >= 2 or exit_count >= 2:
        s -= 1  # multiple institutions reducing

    # Total institutional concentration (0-2)
    tip = result.get("total_inst_pct", 0)
    if tip > 10:
        s += 2
    elif tip > 5:
        s += 1
    elif tip > 2:
        s += 0.5

    return max(1, min(10, s))


def get_institutional_summary(result: Optional[Dict]) -> str:
    """Get a human-readable summary of institutional holdings."""
    if result is None:
        return "无数据"

    parts = []
    if result["huijin"]:
        parts.append("汇金")
    if result["shebao"]:
        parts.append("社保")
    if result["xianzi"]:
        parts.append("险资")
    if result["northbound"]:
        parts.append("北向")
    if result["qfii"]:
        parts.append("QFII")

    if not parts:
        fund_count = result.get("fund_count", 0)
        if fund_count > 0:
            return f"基金({fund_count}只)"
        return "机构少"

    changes = result.get("changes", {})
    new_count = len(changes.get("new", []))
    inc_count = len(changes.get("increased", []))
    dec_count = len(changes.get("decreased", []))

    arrow = ""
    if new_count > 0 or inc_count > 0:
        arrow = "↑"
    elif dec_count > 0:
        arrow = "↓"

    return f"{'/'.join(parts)}{arrow}"


# ── Dependency check ──────────────────────────────────────────────────

def check_deps() -> bool:
    """Check if akshare is available."""
    try:
        import akshare
        return True
    except ImportError:
        return False


import pandas as pd

# ── CLI test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_stocks = [
        ("sh.600519", "贵州茅台"),
        ("sh.688981", "中芯国际"),
        ("sz.300750", "宁德时代"),
        ("sh.600036", "招商银行"),
        ("sh.600276", "恒瑞医药"),
        ("sh.601318", "中国平安"),
    ]
    for code, name in test_stocks:
        result = fetch_institutional(code)
        score = score_institutional(result)
        summary = get_institutional_summary(result)
        changes = result.get("changes", {}) if result else {}
        new_cnt = len(changes.get("new", []))
        inc_cnt = len(changes.get("increased", []))
        dec_cnt = len(changes.get("decreased", []))
        exit_cnt = len(changes.get("exited", []))
        print(f"{name:6s} | 维度={score:.0f} | {summary:20s} | "
              f"汇金={result['huijin'] if result else '?'} "
              f"社保={result['shebao'] if result else '?'} "
              f"险资={result['xianzi'] if result else '?'} "
              f"北向={result['northbound'] if result else '?'} "
              f"QFII={result['qfii'] if result else '?'} | "
              f"新进={new_cnt} 增={inc_cnt} 减={dec_cnt} 退={exit_cnt} | "
              f"机构占比={result['total_inst_pct'] if result else '?'}%")
