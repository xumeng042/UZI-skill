#!/usr/bin/env python3
"""
A股 PE×净利润 九宫格选股系统
基于"市值 = 市盈率 × 净利润"投资框架

Usage:
  streamlit run pe_np_screener.py
"""

import time
import io
import signal
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import streamlit as st

HAS_AKSHARE = False  # Resolved lazily on first use

# ══════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════

ZONE_MAP = {
    ("↑pe", "↑np"): {"zone": 3,  "label": "增长但高估",    "stars": 3, "risk": "中",  "desc": "业绩增长但PE已高"},
    ("↑pe", "→np"): {"zone": 6,  "label": "情绪下跌风险", "stars": 1, "risk": "高",  "desc": "PE高且利润不涨"},
    ("↑pe", "↓np"): {"zone": 9,  "label": "戴维斯双杀",   "stars": 0, "risk": "极高","desc": "业绩估值双杀"},
    ("→pe", "↑np"): {"zone": 2,  "label": "利润驱动上涨", "stars": 4, "risk": "低",  "desc": "赚企业成长的钱"},
    ("→pe", "→np"): {"zone": 5,  "label": "平淡无奇",     "stars": 2, "risk": "低",  "desc": "缺乏催化剂"},
    ("→pe", "↓np"): {"zone": 8,  "label": "利润下滑下跌", "stars": 1, "risk": "中高","desc": "利润下滑缺支撑"},
    ("↓pe", "↑np"): {"zone": 1,  "label": "戴维斯双击",   "stars": 5, "risk": "极低","desc": "最佳投资机会"},
    ("↓pe", "→np"): {"zone": 4,  "label": "估值修复潜力", "stars": 4, "risk": "低",  "desc": "赚估值修复的钱"},
    ("↓pe", "↓np"): {"zone": 7,  "label": "价值陷阱风险", "stars": 2, "risk": "中高","desc": "需判断是否周期底"},
}

ZONE_COLORS = {
    1: ("#00d4aa", "#0a2a1f"),  # 双击 green
    2: ("#22c55e", "#0a2a15"),  # 利润 green
    3: ("#f59e0b", "#2a2005"),  # 增长但高估 amber
    4: ("#3b82f6", "#0a1a35"),  # 修复 blue
    5: ("#6b7280", "#1a1c20"),  # 平淡 gray
    6: ("#f97316", "#2a1505"),  # 情绪 orange
    7: ("#ef4444", "#2a0a0a"),  # 陷阱 red
    8: ("#dc2626", "#2a0808"),  # 下滑 red
    9: ("#991b1b", "#1a0505"),  # 双杀 dark-red
}

INDUSTRY_PE_FALLBACK = {
    "银行": 6, "非银金融": 12, "房地产": 10, "建筑装饰": 10,
    "钢铁": 12, "煤炭": 8, "石油石化": 10, "基础化工": 18,
    "有色金属": 20, "建筑材料": 15, "机械设备": 25, "电力设备": 25,
    "汽车": 22, "家用电器": 15, "食品饮料": 25, "纺织服饰": 18,
    "医药生物": 30, "电子": 35, "计算机": 40, "通信": 20,
    "传媒": 25, "农林牧渔": 22, "国防军工": 45, "公用事业": 15,
    "交通运输": 15, "商贸零售": 20, "社会服务": 28, "环保": 18,
    "轻工制造": 18, "美容护理": 30,
}


# ══════════════════════════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════════════════════════

@contextmanager
def timeout_guard(seconds: int):
    def _handler(signum, frame):
        raise TimeoutError(f"Query timed out after {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _f(row, col, default=0.0):
    try:
        v = row[col]
        return float(v) if pd.notna(v) and v != "" else default
    except (KeyError, ValueError, TypeError):
        return default


# ══════════════════════════════════════════════════════════════════════════
# Data Fetching (baostock)
# ══════════════════════════════════════════════════════════════════════════

def fetch_universe() -> pd.DataFrame:
    import baostock as bs
    bs.login()
    try:
        rs = bs.query_stock_basic()
        df = rs.get_data()
        if df is None or df.empty:
            return pd.DataFrame(columns=["code", "code_name"])
        if "type" in df.columns:
            df = df[df["type"] == "1"].copy()
        return df
    finally:
        bs.logout()


def filter_basic(code: str, name: str) -> bool:
    for tag in ["ST", "*ST", "PT", "退"]:
        if tag in name:
            return False
    if name.startswith("N") or name.startswith("C"):
        return False
    return True


def fetch_one_stock_fast(bs_mod, code: str, name: str) -> Optional[Dict[str, Any]]:
    m = {
        "code": code.replace("sh.", "SH").replace("sz.", "SZ").replace("bj.", "BJ"),
        "name": name,
    }
    try:
        _fetch_financials(bs_mod, code, m)
        _fetch_balance(bs_mod, code, m)
        _fetch_industry_info(bs_mod, code, m)
        _fetch_pe_from_kline(bs_mod, code, m)
        _set_fast_defaults(m)
        return m
    except TimeoutError:
        return None
    except Exception as e:
        err = str(e)
        if "接收数据异常" in err or "timeout" in err.lower():
            return None
        return None


def _fetch_financials(bs_mod, code, m):
    h = {"rev": [], "profit": [], "roe": [], "gm": []}
    for year in [2021, 2022, 2023, 2024, 2025]:
        rs = bs_mod.query_profit_data(code=code, year=year, quarter=4)
        if rs.error_code != "0":
            continue
        df = rs.get_data()
        if df.empty:
            continue
        r = df.iloc[-1]
        h["rev"].append(_f(r, "MBRevenue"))
        h["profit"].append(_f(r, "netProfit"))
        h["roe"].append(_f(r, "roeAvg") * 100)
        h["gm"].append(_f(r, "gpMargin") * 100)

    revs = pd.Series(h["rev"]).dropna()
    profits = pd.Series(h["profit"]).dropna()
    roes = pd.Series(h["roe"]).dropna()
    gms = pd.Series(h["gm"]).dropna()

    if len(revs) > 0:
        m["revenue_latest"] = round(revs.iloc[-1] / 1e8, 1)
    if len(profits) > 0:
        m["net_profit_latest"] = round(profits.iloc[-1] / 1e8, 1)
    if len(roes) > 0:
        m["roe_latest"] = round(roes.iloc[-1], 1)
    if len(gms) > 0:
        m["gross_margin_latest"] = round(gms.iloc[-1], 1)

    if len(revs) >= 3 and revs.iloc[-3] > 0:
        m["revenue_cagr_3y"] = round(((revs.iloc[-1] / revs.iloc[-3]) ** (1 / 3) - 1) * 100, 1)
    if len(revs) >= 2 and revs.iloc[-2] > 0:
        m["revenue_growth"] = round((revs.iloc[-1] / revs.iloc[-2] - 1) * 100, 1)

    if len(profits) >= 3 and profits.iloc[-3] > 0:
        r = profits.iloc[-1] / profits.iloc[-3]
        if r > 0:
            m["profit_cagr_3y"] = round((r ** (1 / 3) - 1) * 100, 1)
    if len(profits) >= 2 and profits.iloc[-2] > 0:
        m["profit_growth"] = round((profits.iloc[-1] / profits.iloc[-2] - 1) * 100, 1)

    if len(profits) >= 4 and profits.iloc[-2] > 0 and profits.iloc[-3] > 0:
        gl = (profits.iloc[-1] / profits.iloc[-2] - 1) * 100
        gp = (profits.iloc[-2] / profits.iloc[-3] - 1) * 100
        m["earnings_growth_latest"] = round(gl, 1)
        m["earnings_growth_prior"] = round(gp, 1)
        m["earnings_accel"] = round(gl - gp, 1)

    consecutive = 0
    for p in reversed(profits):
        if p > 0:
            consecutive += 1
        else:
            break
    m["consecutive_profit_years"] = consecutive


def _fetch_balance(bs_mod, code, m):
    for (y, q) in [(2025, 4), (2024, 4), (2023, 4)]:
        rs = bs_mod.query_balance_data(code=code, year=y, quarter=q)
        if rs.error_code != "0":
            continue
        df = rs.get_data()
        if df.empty:
            continue
        r = df.iloc[-1]
        m["current_ratio"] = round(_f(r, "currentRatio"), 2)
        ate = _f(r, "assetToEquity")
        if ate > 1:
            m["debt_ratio"] = round((1 - 1 / ate) * 100, 1)
        return


def _fetch_industry_info(bs_mod, code, m):
    try:
        rs = bs_mod.query_stock_industry(code=code)
        if rs.error_code == "0":
            df = rs.get_data()
            if not df.empty:
                ind = df.iloc[0].get("industry", "")
                m["industry"] = str(ind) if pd.notna(ind) else ""
                return
    except Exception:
        pass
    m["industry"] = ""


def _fetch_pe_from_kline(bs_mod, code, m):
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
        rs = bs_mod.query_history_k_data_plus(
            code, "date,close,peTTM,pbMRQ",
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="3"
        )
        if rs.error_code != "0":
            return
        df = rs.get_data()
        if df.empty:
            return
        pes = pd.to_numeric(df["peTTM"], errors="coerce").dropna()
        pbs = pd.to_numeric(df["pbMRQ"], errors="coerce").dropna()
        closes = pd.to_numeric(df["close"], errors="coerce").dropna()
        if len(pes) > 0:
            m["pe_ttm"] = round(pes.iloc[-1], 2)
            m["pe_60d_min"] = round(pes.min(), 2)
            m["pe_60d_max"] = round(pes.max(), 2)
            m["pe_60d_median"] = round(pes.median(), 2)
            if pes.max() > pes.min():
                m["pe_60d_pct"] = round((pes.iloc[-1] - pes.min()) / (pes.max() - pes.min()) * 100, 1)
            else:
                m["pe_60d_pct"] = 50.0
        if len(pbs) > 0:
            m["pb"] = round(pbs.iloc[-1], 2)
        if len(closes) > 0:
            m["close_price"] = round(closes.iloc[-1], 2)
    except Exception:
        pass


def _set_fast_defaults(m):
    m.setdefault("revenue_latest", 0)
    m.setdefault("net_profit_latest", 0)
    m.setdefault("roe_latest", 0)
    m.setdefault("gross_margin_latest", 0)
    m.setdefault("revenue_cagr_3y", 0)
    m.setdefault("revenue_growth", 0)
    m.setdefault("profit_cagr_3y", 0)
    m.setdefault("profit_growth", 0)
    m.setdefault("earnings_growth_latest", 0)
    m.setdefault("earnings_growth_prior", 0)
    m.setdefault("earnings_accel", 0)
    m.setdefault("consecutive_profit_years", 0)
    m.setdefault("debt_ratio", 50)
    m.setdefault("current_ratio", 1.0)
    m.setdefault("industry", "")
    m.setdefault("pe_ttm", 0)
    m.setdefault("pe_60d_min", 0)
    m.setdefault("pe_60d_max", 0)
    m.setdefault("pe_60d_median", 0)
    m.setdefault("pe_60d_pct", 50)
    m.setdefault("pb", 0)
    m.setdefault("close_price", 0)
    m.setdefault("pe_quantile_5y", 50)


# ══════════════════════════════════════════════════════════════════════════
# PE Percentile Enrichment (akshare)
# ══════════════════════════════════════════════════════════════════════════

def enrich_pe_percentile(stock_data: Dict[str, Any]) -> Dict[str, Any]:
    if not HAS_AKSHARE:
        return stock_data
    import akshare as ak
    code = stock_data.get("code", "")
    symbol = code.replace("SH", "").replace("SZ", "").replace("BJ", "")
    if not symbol:
        return stock_data
    try:
        df = ak.stock_zh_valuation_baidu(symbol=symbol, indicator="市盈率(TTM)", period="近五年")
        if df is None or df.empty:
            stock_data["pe_quantile_5y"] = stock_data.get("pe_60d_pct", 50)
            stock_data["pe_quantile_source"] = "60d_proxy"
            return stock_data
        pes = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
        if len(pes) < 10:
            stock_data["pe_quantile_5y"] = stock_data.get("pe_60d_pct", 50)
            stock_data["pe_quantile_source"] = "60d_proxy"
            return stock_data
        cur_pe = stock_data.get("pe_ttm", 0)
        if cur_pe <= 0:
            cur_pe = pes.iloc[-1]
        pct = (pes < cur_pe).sum() / len(pes) * 100
        stock_data["pe_quantile_5y"] = round(pct, 1)
        stock_data["pe_quantile_source"] = "5y_baidu"
        stock_data["pe_5y_min"] = round(pes.min(), 2)
        stock_data["pe_5y_max"] = round(pes.max(), 2)
        stock_data["pe_5y_median"] = round(pes.median(), 2)
        stock_data["pe_5y_samples"] = len(pes)
    except Exception:
        stock_data["pe_quantile_5y"] = stock_data.get("pe_60d_pct", 50)
        stock_data["pe_quantile_source"] = "60d_proxy"
    return stock_data


# ══════════════════════════════════════════════════════════════════════════
# Core: 9-Zone Classification & Scoring
# ══════════════════════════════════════════════════════════════════════════

def classify_pe_direction(m: Dict[str, Any]) -> str:
    pct = m.get("pe_quantile_5y", 50)
    if pct <= 30:
        return "↓pe"
    elif pct <= 70:
        return "→pe"
    else:
        return "↑pe"


def classify_np_direction(m: Dict[str, Any]) -> str:
    growth = m.get("profit_growth", 0)
    cagr = m.get("profit_cagr_3y", 0)
    accel = m.get("earnings_accel", 0)
    if growth > 10 and cagr > 5:
        return "↑np"
    elif growth > 10:
        return "↑np"
    elif growth < -5 and cagr < 0:
        return "↓np"
    elif growth < -5:
        return "↓np"
    elif growth >= -5 and accel > 3:
        return "↑np"
    elif growth >= -5:
        return "→np"
    else:
        return "↓np"


def classify_zone(m: Dict[str, Any]) -> Dict[str, Any]:
    pe_dir = classify_pe_direction(m)
    np_dir = classify_np_direction(m)
    zone_info = ZONE_MAP.get((pe_dir, np_dir), {
        "zone": 5, "label": "难以判断", "stars": 2, "risk": "中",
        "desc": "数据不足"
    })
    m["pe_direction"] = pe_dir
    m["np_direction"] = np_dir
    m["zone"] = zone_info["zone"]
    m["zone_label"] = zone_info["label"]
    m["zone_stars"] = zone_info["stars"]
    m["zone_risk"] = zone_info["risk"]
    m["zone_desc"] = zone_info["desc"]

    if zone_info["zone"] in [1, 2]:
        m["investor_type"] = "第四类 · 长期价值投资者"
        m["money_type"] = "赚企业成长的钱（净利润增长）"
        if zone_info["zone"] == 1:
            m["money_type"] = "赚企业成长的钱 + 估值修复的钱（戴维斯双击）"
    elif zone_info["zone"] == 4:
        m["investor_type"] = "第三类 · 逆向价值型"
        m["money_type"] = "赚估值修复的钱（市盈率从低到正常）"
    elif zone_info["zone"] == 3:
        m["investor_type"] = "第四类（需警惕估值）"
        m["money_type"] = "赚企业成长的钱，但需消化高估值"
    elif zone_info["zone"] == 7:
        m["investor_type"] = "第三类（需警惕价值陷阱）"
        m["money_type"] = "估值低但利润在降，需判断是否周期性底部"
    elif zone_info["zone"] in [6, 9]:
        m["investor_type"] = "不建议参与"
        m["money_type"] = "大概率亏钱——建议回避"
    else:
        m["investor_type"] = "观望"
        m["money_type"] = "方向不明确，等待信号"
    return m


def score_stock_pe_np(m: Dict[str, Any]) -> Dict[str, Any]:
    eq_score = 0.0
    pg = m.get("profit_growth", 0)
    if pg > 30:       eq_score += 20
    elif pg > 15:     eq_score += 16
    elif pg > 10:     eq_score += 12
    elif pg > 5:      eq_score += 8
    elif pg > 0:      eq_score += 4
    cagr = m.get("profit_cagr_3y", 0)
    if cagr > 20:     eq_score += 10
    elif cagr > 10:   eq_score += 7
    elif cagr > 5:    eq_score += 4
    elif cagr > 0:    eq_score += 2
    accel = m.get("earnings_accel", 0)
    if accel > 5:     eq_score += 5
    elif accel > 2:   eq_score += 3
    elif accel > 0:   eq_score += 1
    roe = m.get("roe_latest", 0)
    if roe >= 20:     eq_score += 5
    elif roe >= 15:   eq_score += 4
    elif roe >= 10:   eq_score += 3
    elif roe >= 8:    eq_score += 2
    elif roe > 0:     eq_score += 1
    m["score_earnings"] = round(eq_score, 1)

    val_score = 0.0
    pct = m.get("pe_quantile_5y", 50)
    if pct <= 10:     val_score += 25
    elif pct <= 20:   val_score += 22
    elif pct <= 30:   val_score += 18
    elif pct <= 40:   val_score += 12
    elif pct <= 50:   val_score += 8
    elif pct <= 60:   val_score += 4
    elif pct <= 70:   val_score += 2
    pe = m.get("pe_ttm", 0)
    if 5 < pe < 10:       val_score += 10
    elif 10 <= pe < 15:   val_score += 8
    elif 15 <= pe < 20:   val_score += 6
    elif 20 <= pe < 25:   val_score += 4
    elif 25 <= pe < 30:   val_score += 2
    pb = m.get("pb", 0)
    if 0 < pb < 1.0:  val_score += 5
    elif pb < 1.5:    val_score += 3
    elif pb < 2.0:    val_score += 1
    m["score_valuation"] = round(val_score, 1)

    fh_score = 0.0
    debt = m.get("debt_ratio", 50)
    if debt < 30:     fh_score += 10
    elif debt < 45:   fh_score += 7
    elif debt < 60:   fh_score += 4
    elif debt < 70:   fh_score += 2
    cr = m.get("current_ratio", 1.0)
    if cr > 2.0:      fh_score += 5
    elif cr > 1.5:    fh_score += 3
    elif cr > 1.0:    fh_score += 1
    cons = m.get("consecutive_profit_years", 0)
    if cons >= 5:     fh_score += 5
    elif cons >= 3:   fh_score += 3
    elif cons >= 2:   fh_score += 1
    m["score_health"] = round(fh_score, 1)

    m["score_total"] = round(eq_score + val_score + fh_score, 1)
    return m


def filter_financial_quality(m: Dict[str, Any]) -> Optional[str]:
    if m.get("net_profit_latest", 0) <= 0:
        return "净利润为负"
    if m.get("roe_latest", 0) < 5:
        return f"ROE过低({m['roe_latest']:.1f}%)"
    if m.get("revenue_latest", 0) < 1.0:
        return f"营收过低({m['revenue_latest']:.1f}亿)"
    if m.get("debt_ratio", 0) > 80:
        return f"负债过高({m['debt_ratio']:.0f}%)"
    if m.get("pe_ttm", 0) <= 0:
        return "PE数据缺失"
    return None


# ══════════════════════════════════════════════════════════════════════════
# Global CSS
# ══════════════════════════════════════════════════════════════════════════

GLOBAL_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root overrides ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #e8edf5;
}

/* ── Hide Streamlit chrome ── */
[data-testid="stDecoration"], #MainMenu, footer, header {
    visibility: hidden;
    height: 0;
}

/* ── Main content area ── */
[data-testid="stAppViewContainer"] > .main {
    background: linear-gradient(180deg, #0a0e17 0%, #0d1320 100%);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1220 0%, #0a0f1a 100%);
    border-right: 1px solid #1e2a3f;
}
[data-testid="stSidebar"] .st-emotion-cache-1avcm0n {
    background: transparent;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    border-radius: 8px;
    padding: 10px 24px;
    transition: all 0.2s ease;
    letter-spacing: 0.02em;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00d4aa 0%, #00a88a 100%);
    color: #0a0e17;
    border: none;
    box-shadow: 0 2px 8px rgba(0,212,170,0.25);
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 16px rgba(0,212,170,0.4);
    transform: translateY(-1px);
}

/* ── Select boxes & sliders ── */
[data-testid="stSelectbox"], .stSlider {
    font-family: 'Inter', sans-serif;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2a3f;
    border-radius: 10px;
    overflow: hidden;
}
[data-testid="stDataFrame"] th {
    background: #131b2a !important;
    color: #8896b0 !important;
    font-weight: 600;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 10px 12px !important;
    border-bottom: 2px solid #1e2a3f !important;
}
[data-testid="stDataFrame"] td {
    background: #0d1320 !important;
    color: #e8edf5 !important;
    font-size: 0.82rem;
    padding: 8px 12px !important;
    border-bottom: 1px solid #131b2a !important;
}
[data-testid="stDataFrame"] tr:hover td {
    background: #131b2a !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 1px solid #1e2a3f;
    border-radius: 10px;
    background: #0d1320;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #00d4aa, #3b82f6);
    border-radius: 4px;
}

/* ── Horizontal divider ── */
hr {
    border-color: #1e2a3f;
    margin: 24px 0;
}

/* ══════ Custom component styles ══════ */

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, #0a1628 0%, #111d35 50%, #0a0e17 100%);
    border: 1px solid #1e2a3f;
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0,212,170,0.06) 0%, transparent 70%);
}
.hero-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e8edf5;
    margin: 0;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    font-size: 0.88rem;
    color: #5a6980;
    margin-top: 6px;
    font-weight: 400;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    min-width: 140px;
    background: #111827;
    border: 1px solid #1e2a3f;
    border-radius: 10px;
    padding: 14px 18px;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    border-color: #2d4a7a;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #5a6980;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #e8edf5;
}
.metric-delta {
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: 2px;
}

/* Zone matrix 3x3 */
.zone-matrix {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    grid-template-rows: repeat(3, 1fr);
    gap: 6px;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #1e2a3f;
}
.zone-cell {
    padding: 14px 10px;
    text-align: center;
    font-weight: 600;
    font-size: 0.8rem;
    transition: all 0.2s;
}
.zone-cell:hover {
    filter: brightness(1.2);
    transform: scale(1.02);
}
.zone-num {
    font-size: 1.1rem;
    font-weight: 800;
}
.zone-count {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 2px 0;
}

/* Stock card */
.stock-card {
    background: #111827;
    border: 1px solid #1e2a3f;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 6px 0;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stock-card:hover {
    border-color: #2d4a7a;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
}
.stock-card .name {
    font-size: 1.05rem;
    font-weight: 700;
    color: #e8edf5;
}
.stock-card .code {
    font-size: 0.78rem;
    color: #5a6980;
    margin-left: 8px;
}
.stock-card .score {
    font-size: 1.3rem;
    font-weight: 700;
}
.stock-card .meta {
    font-size: 0.78rem;
    color: #8896b0;
    margin-top: 4px;
}
.stock-card .money-type {
    font-size: 0.78rem;
    margin-top: 4px;
    opacity: 0.9;
}

/* Score bar */
.score-bar-wrap {
    margin: 6px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.score-bar-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #5a6980;
    width: 60px;
    text-align: right;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.score-bar-track {
    flex: 1;
    height: 6px;
    background: #1e2a3f;
    border-radius: 3px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.4s ease;
}
.score-bar-val {
    font-size: 0.72rem;
    font-weight: 600;
    color: #8896b0;
    width: 36px;
}

/* Welcome cards */
.welcome-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin: 20px 0;
}
.welcome-card {
    background: linear-gradient(135deg, #111827 0%, #0d1320 100%);
    border: 1px solid #1e2a3f;
    border-radius: 12px;
    padding: 24px 20px;
    text-align: center;
}
.welcome-card .icon {
    font-size: 2rem;
    margin-bottom: 10px;
}
.welcome-card h3 {
    font-size: 1rem;
    font-weight: 700;
    color: #e8edf5;
    margin: 0 0 6px 0;
}
.welcome-card p {
    font-size: 0.8rem;
    color: #8896b0;
    margin: 0;
    line-height: 1.5;
}

/* Summary bar */
.summary-bar {
    background: #111827;
    border: 1px solid #1e2a3f;
    border-radius: 10px;
    padding: 14px 20px;
    font-size: 0.82rem;
    color: #5a6980;
    margin: 12px 0;
}

/* Section title */
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e8edf5;
    margin: 24px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e2a3f;
}

/* Tab-style zone row */
.zone-card-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
}
.zone-card-mini {
    background: #111827;
    border: 1px solid #1e2a3f;
    border-radius: 10px;
    padding: 16px;
}
.zone-card-mini h4 {
    font-size: 0.9rem;
    font-weight: 700;
    margin: 0 0 4px 0;
}
.zone-card-mini .desc {
    font-size: 0.75rem;
    color: #5a6980;
    margin-bottom: 10px;
}

/* Color utilities */
.c-green  { color: #00d4aa; }
.c-blue   { color: #3b82f6; }
.c-gold   { color: #f0b90b; }
.c-red    { color: #ff4757; }
.c-orange { color: #f97316; }
.c-gray   { color: #6b7280; }
.bg-green  { background: #00d4aa; }
.bg-blue   { background: #3b82f6; }
.bg-gold   { background: #f0b90b; }
.bg-red    { background: #ff4757; }

</style>
"""

# ══════════════════════════════════════════════════════════════════════════
# Streamlit Page Config
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="PE×NP 九宫格选股",
    page_icon="🎯",
    layout="wide",
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── Session state ──
for key in ["results", "stats", "last_run_at"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ══════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 4px 0;">
        <div style="font-size:1.1rem;font-weight:700;color:#e8edf5;">PE × NP 九宫格</div>
        <div style="font-size:0.75rem;color:#5a6980;">市值 = 市盈率 × 净利润</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    max_options = {
        "快速测试 (50只)": 50,
        "中等 (200只)": 200,
        "标准 (500只)": 500,
        "深度 (1000只)": 1000,
        "全量 (所有A股)": 0,
    }
    max_choice = st.selectbox("扫描范围", list(max_options.keys()), index=1)
    max_stocks = max_options[max_choice]

    top_n = st.slider("输出前 N 名", 10, 50, 30, 5)

    rate_options = {"慢速 3/s": 3.0, "标准 5/s": 5.0, "快速 8/s": 8.0, "极速 15/s": 15.0}
    rate_choice = st.selectbox("查询速度", list(rate_options.keys()), index=1)
    rate_limit = rate_options[rate_choice]

    enrich_pe_pct = st.checkbox("PE百分位增强 (akshare)", value=True,
                                help="获取5年PE百分位。关闭则用60日近似。")

    st.divider()

    run_clicked = st.button("开始九宫格选股", type="primary", use_container_width=True)

    st.divider()

    with st.expander("九宫格框架"):
        st.markdown("""
|  | PE↓ | PE→ | PE↑ |
|---|---|---|---|
| **NP↑** | ①双击⭐5 | ②利润驱动⭐4 | ③增长高估⭐3 |
| **NP→** | ④估值修复⭐4 | ⑤平淡⭐2 | ⑥情绪下跌⭐1 |
| **NP↓** | ⑦价值陷阱⭐2 | ⑧利润下滑⭐1 | ⑨双杀☆ |

PE%≤30→低位↓, 30-70→中→, >70→高位↑
NP增速>10%→增长↑, -5~10%→稳→, <-5%→下滑↓
        """)

    with st.expander("投资逻辑"):
        st.markdown("""
**四类参与者：**
1. 随波逐流型 → 亏钱主力
2. 短线情绪交易 → 赚PE波动(极难)
3. **逆向价值型** → 赚估值修复
4. **长期价值型** → 赚企业成长

**目标：找③+④类机会**
- 优先 ①区(双击) ②区(利润驱动)
- 关注 ④区(估值修复)
- 警惕 ⑦区(陷阱) ⑨区(双杀)
        """)

    with st.expander("评分体系"):
        st.markdown("""
| 维度 | 权重 |
|---|---|
| 盈利质量 | 40% |
| 估值位置 | 40% |
| 财务健康 | 20% |

总分 0-100，越高越符合投资逻辑
        """)


# ══════════════════════════════════════════════════════════════════════════
# Cached universe
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def cached_universe():
    return fetch_universe()


# ══════════════════════════════════════════════════════════════════════════
# Screening Runner
# ══════════════════════════════════════════════════════════════════════════

def run_screening(max_stocks, top_n, rate_limit, enrich_pe):
    t0 = time.monotonic()
    status_box = st.empty()
    status_box.info("正在获取A股全市场列表...")

    df = cached_universe()
    candidates = []
    for _, row in df.iterrows():
        code = str(row.iloc[0]).strip()
        name = str(row.iloc[1]).strip()
        if filter_basic(code, name):
            candidates.append((code, name))
    if max_stocks > 0:
        candidates = candidates[:max_stocks]

    status_box.info(f"全市场 {len(df)} 只 → 初筛后 {len(candidates)} 只，开始逐只分析...")

    prog = st.progress(0, text=f"0 / {len(candidates)}")
    stat_line = st.empty()

    import baostock as bs
    bs.login()
    try:
        results = []
        failures = 0
        filtered = 0
        for i, (code, name) in enumerate(candidates):
            frac = (i + 1) / len(candidates)
            prog.progress(frac, text=f"{i+1}/{len(candidates)} — {name}")
            time.sleep(1.0 / rate_limit)
            m = fetch_one_stock_fast(bs, code, name)
            if m is None:
                failures += 1
                continue
            fail_reason = filter_financial_quality(m)
            if fail_reason:
                filtered += 1
                continue
            classify_zone(m)
            score_stock_pe_np(m)
            results.append(m)
            if (i + 1) % 25 == 0:
                elapsed = time.monotonic() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                stat_line.text(f"已处理: {i+1} | 通过: {len(results)} | 过滤: {filtered} | 失败: {failures} | 速率: {rate:.1f}/s")
    finally:
        bs.logout()

    elapsed = time.monotonic() - t0
    prog.empty()
    stat_line.empty()
    status_box.empty()

    results.sort(key=lambda r: r["score_total"], reverse=True)

    if enrich_pe and len(results) > 0:
        global HAS_AKSHARE
        try:
            import akshare  # noqa
            HAS_AKSHARE = True
        except ImportError:
            HAS_AKSHARE = False
        if HAS_AKSHARE:
            enrich_n = min(len(results), max(top_n, 30))
            enrich_status = st.empty()
            enrich_status.info(f"PE历史百分位数据获取中 — 前 {enrich_n} 只...")
            for i in range(enrich_n):
                enrich_status.info(f"PE百分位: {i+1}/{enrich_n} — {results[i]['name']}")
                time.sleep(0.5)
                enrich_pe_percentile(results[i])
                classify_zone(results[i])
                score_stock_pe_np(results[i])
            enrich_status.empty()
            results.sort(key=lambda r: r["score_total"], reverse=True)

    results = results[:top_n]
    stats = {
        "universe": len(df), "t1_passed": len(candidates),
        "t2_passed": len(results), "elapsed": round(elapsed, 1),
        "failures": failures, "filtered": filtered,
    }
    return results, stats


def build_results_df(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(results):
        rows.append({
            "#": i + 1,
            "代码": r["code"],
            "名称": r["name"],
            "总分": r["score_total"],
            "九宫格": f"{r['zone']}区·{r['zone_label']}",
            "⭐": "★" * r["zone_stars"],
            "赚钱逻辑": r["money_type"],
            "PE(TTM)": r["pe_ttm"],
            "PE百分位": r.get("pe_quantile_5y", 50),
            "PB": r.get("pb", 0),
            "净利增速%": r.get("profit_growth", 0),
            "3年CAGR%": r.get("profit_cagr_3y", 0),
            "增速加速度": r.get("earnings_accel", 0),
            "ROE%": r.get("roe_latest", 0),
            "负债率%": r.get("debt_ratio", 0),
            "营收(亿)": r.get("revenue_latest", 0),
            "盈利得分": r["score_earnings"],
            "估值得分": r["score_valuation"],
            "健康得分": r["score_health"],
            "行业": r.get("industry", ""),
            "连续盈利年": r.get("consecutive_profit_years", 0),
            "风险等级": r["zone_risk"],
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# Render: Welcome
# ══════════════════════════════════════════════════════════════════════════

def render_welcome():
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">市值 = 市盈率 × 净利润</div>
        <div class="hero-subtitle">股价涨跌只有两个变量：公司赚不赚钱 · 市场给不给估值 · 九种组合三种能赚钱</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="welcome-grid">', unsafe_allow_html=True)

    st.markdown("""
    <div class="welcome-card">
        <div class="icon">🔍</div>
        <h3>PE百分位定位</h3>
        <p>当前PE处于历史什么位置？<br>低位→估值修复空间<br>高位→估值收缩风险</p>
    </div>
    <div class="welcome-card">
        <div class="icon">📈</div>
        <h3>净利润趋势判断</h3>
        <p>公司赚钱能力在增强还是减弱？<br>增长→业绩驱动上涨<br>下滑→基本面恶化</p>
    </div>
    <div class="welcome-card">
        <div class="icon">🎯</div>
        <h3>九宫格精准定位</h3>
        <p>每只股票归入9个区域之一<br>目标：第三类+第四类机会<br>赚看得懂的钱</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-title">能赚钱的三种模式</div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="stock-card" style="border-left:3px solid #00d4aa;">
            <div class="name" style="color:#00d4aa;">① 戴维斯双击</div>
            <div class="meta">PE低位 + 利润增长</div>
            <div class="money-type" style="color:#00d4aa;">赚PE修复 + 利润成长</div>
            <div style="font-size:0.75rem;color:#5a6980;margin-top:6px;">最理想的投资机会 · 双击暴利</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="stock-card" style="border-left:3px solid #22c55e;">
            <div class="name" style="color:#22c55e;">② 利润驱动上涨</div>
            <div class="meta">PE合理 + 利润增长</div>
            <div class="money-type" style="color:#22c55e;">赚企业成长的钱</div>
            <div style="font-size:0.75rem;color:#5a6980;margin-top:6px;">第四类投资者 · 稳健确定性</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="stock-card" style="border-left:3px solid #3b82f6;">
            <div class="name" style="color:#3b82f6;">④ 估值修复潜力</div>
            <div class="meta">PE低位 + 利润稳定</div>
            <div class="money-type" style="color:#3b82f6;">赚估值修复的钱</div>
            <div style="font-size:0.75rem;color:#5a6980;margin-top:6px;">第三类投资者 · 逆向价值</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-title">大部分散户亏钱的模式</div>
    """, unsafe_allow_html=True)

    w1, w2, w3 = st.columns(3)
    with w1:
        st.markdown("""
        <div class="stock-card" style="border-left:3px solid #f97316; opacity:0.8;">
            <div class="name" style="color:#f97316;">⑥ 情绪炒作追高</div>
            <div class="meta">PE高位 + 利润不涨</div>
            <div style="font-size:0.75rem;color:#f97316;margin-top:4px;">追涨杀跌→情绪退潮即亏损</div>
        </div>
        """, unsafe_allow_html=True)
    with w2:
        st.markdown("""
        <div class="stock-card" style="border-left:3px solid #dc2626; opacity:0.8;">
            <div class="name" style="color:#dc2626;">⑨ 戴维斯双杀</div>
            <div class="meta">PE高位 + 利润下滑</div>
            <div style="font-size:0.75rem;color:#dc2626;margin-top:4px;">亏钱最快的方式→坚决回避</div>
        </div>
        """, unsafe_allow_html=True)
    with w3:
        st.markdown("""
        <div class="stock-card" style="border-left:3px solid #ef4444; opacity:0.8;">
            <div class="name" style="color:#ef4444;">⑦ 价值陷阱</div>
            <div class="meta">PE低位 + 利润下滑</div>
            <div style="font-size:0.75rem;color:#ef4444;margin-top:4px;">看似便宜→实则基本面恶化</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; margin-top:32px; color:#5a6980; font-size:0.85rem;">
        左侧设置参数，点击 <b style="color:#00d4aa;">开始九宫格选股</b> 开始扫描
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# Render: Results
# ══════════════════════════════════════════════════════════════════════════

def render_results(results, stats):
    df = build_results_df(results)

    st.toast(f"选股完成！入围 {len(results)} 只，耗时 {stats['elapsed']:.0f}s", icon="✅")

    # ── Hero header with key stats ──
    zone1_count = sum(1 for r in results if r["zone"] == 1)
    zone2_count = sum(1 for r in results if r["zone"] == 2)
    zone4_count = sum(1 for r in results if r["zone"] == 4)
    avg_score = sum(r["score_total"] for r in results) / len(results)

    st.markdown(f"""
    <div class="hero-header">
        <div class="hero-title">扫描完成 · {len(results)} 只股票入围</div>
        <div class="hero-subtitle">耗时 {stats['elapsed']:.0f}s · 全市场 {stats['universe']} 只 → 入围 {stats['t2_passed']} 只</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric cards ──
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    mc_data = [
        ("入围股票", f"{len(results)}", "只"),
        ("戴维斯双击", f"{zone1_count}", "①区"),
        ("利润驱动", f"{zone2_count}", "②区"),
        ("估值修复", f"{zone4_count}", "④区"),
        ("平均总分", f"{avg_score:.1f}", "/100"),
        ("总耗时", f"{stats['elapsed']:.0f}", "秒"),
    ]
    for col, (label, value, unit) in zip([m1, m2, m3, m4, m5, m6], mc_data):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-delta" style="color:#5a6980;">{unit}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="summary-bar">
        全市场 {stats['universe']} 只 → 初筛 {stats['t1_passed']} 只 →
        入围 {stats['t2_passed']} 只 | 失败 {stats['failures']} 只 | 过滤 {stats['filtered']} 只
    </div>
    """, unsafe_allow_html=True)

    # ── 3x3 Zone Matrix ──
    st.markdown('<div class="section-title">九宫格分布热力图</div>', unsafe_allow_html=True)

    # Build counts for each cell
    pe_keys = ["↓pe", "→pe", "↑pe"]
    np_keys = ["↑np", "→np", "↓np"]
    zone_ids = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    pe_labels = ["PE 低位 ↓", "PE 中性 →", "PE 高位 ↑"]

    zcols = st.columns(3)
    for ci, (pe_key, pe_label) in enumerate(zip(pe_keys, pe_labels)):
        with zcols[ci]:
            st.markdown(f'<div style="font-size:0.78rem;font-weight:700;color:#8896b0;text-align:center;margin-bottom:4px;">{pe_label}</div>', unsafe_allow_html=True)
            for ri, np_key in enumerate(np_keys):
                count = sum(1 for r in results if r["pe_direction"] == pe_key and r["np_direction"] == np_key)
                zid = zone_ids[ri][ci]
                border_color, bg_color = ZONE_COLORS[zid]
                np_labels = ["NP ↑ 增长", "NP → 稳定", "NP ↓ 下滑"]
                st.markdown(f"""
                <div style="background:{bg_color}; border:1px solid {border_color}40; border-radius:8px; padding:12px 10px; margin:3px 0; text-align:center;">
                    <div style="font-size:0.65rem;color:#5a6980;">{np_labels[ri]}</div>
                    <div style="font-size:1.4rem;font-weight:700;color:{border_color};">{count}</div>
                    <div style="font-size:0.7rem;font-weight:600;color:{border_color};">{zid}区 {ZONE_MAP[(pe_key,np_key)]['label']}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Score bars ──
    st.markdown('<div class="section-title">三维度平均得分</div>', unsafe_allow_html=True)
    dims = [
        ("盈利质量", df["盈利得分"].mean(), 40, "#00d4aa"),
        ("估值位置", df["估值得分"].mean(), 40, "#3b82f6"),
        ("财务健康", df["健康得分"].mean(), 20, "#f0b90b"),
    ]
    for label, val, max_val, color in dims:
        pct = val / max_val * 100
        st.markdown(f"""
        <div class="score-bar-wrap">
            <span class="score-bar-label">{label}</span>
            <div class="score-bar-track">
                <div class="score-bar-fill" style="width:{pct:.0f}%;background:{color};"></div>
            </div>
            <span class="score-bar-val">{val:.1f}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Data table ──
    st.markdown(f'<div class="section-title">TOP {len(results)} 排名</div>', unsafe_allow_html=True)

    def _zone_bg(val):
        s = str(val)
        if "①" in s or "④" in s:
            return "background-color: #0a2a1f; color: #00d4aa; font-weight: 600"
        elif "②" in s:
            return "background-color: #0a2a15; color: #22c55e; font-weight: 600"
        elif "③" in s or "⑦" in s:
            return "background-color: #2a2005; color: #f59e0b"
        elif "⑥" in s or "⑧" in s:
            return "background-color: #2a1505; color: #f97316"
        elif "⑨" in s:
            return "background-color: #1a0505; color: #ef4444; font-weight: 600"
        return ""

    def _pe_pct_bg(val):
        if val <= 20:
            return "background-color: #0a2a1f; color: #00d4aa; font-weight: 600"
        elif val <= 40:
            return "background-color: #0a2a15; color: #22c55e"
        elif val <= 60:
            return "background-color: #2a2005; color: #f0b90b"
        else:
            return "background-color: #2a0808; color: #ef4444"

    def _np_bg(val):
        if val > 15:
            return "background-color: #0a2a1f; color: #00d4aa; font-weight: 600"
        elif val > 5:
            return "background-color: #0a2a15; color: #22c55e"
        elif val > -5:
            return "background-color: #2a2005; color: #f0b90b"
        else:
            return "background-color: #2a0808; color: #ef4444"

    def _score_bg(val):
        if val >= 30:
            return "background-color: #0a2a1f; color: #00d4aa"
        elif val >= 20:
            return "background-color: #0a2a15; color: #22c55e"
        elif val >= 10:
            return "background-color: #2a2005; color: #f0b90b"
        else:
            return "background-color: #2a0808; color: #ef4444"

    score_cols = ["盈利得分", "估值得分", "健康得分"]

    styler = df.style.applymap(_zone_bg, subset=["九宫格"]) \
        .applymap(_pe_pct_bg, subset=["PE百分位"]) \
        .applymap(_np_bg, subset=["净利增速%"]) \
        .applymap(_score_bg, subset=score_cols) \
        .applymap(lambda v: "background-color: #0d1320; color: #e8edf5; font-weight: 700" if True else "", subset=["总分"]) \
        .format({
            "总分": "{:.1f}", "PE(TTM)": "{:.1f}", "PE百分位": "{:.1f}",
            "PB": "{:.2f}", "净利增速%": "{:+.1f}", "3年CAGR%": "{:+.1f}",
            "增速加速度": "{:+.1f}", "ROE%": "{:.1f}", "负债率%": "{:.1f}",
            "盈利得分": "{:.1f}", "估值得分": "{:.1f}", "健康得分": "{:.1f}",
        })

    st.dataframe(styler, use_container_width=True, hide_index=True,
                 column_config={
                     "#": st.column_config.NumberColumn("#", width="small"),
                     "代码": st.column_config.TextColumn("代码", width="small"),
                     "名称": st.column_config.TextColumn("名称", width="small"),
                     "总分": st.column_config.NumberColumn("总分", width="small"),
                     "九宫格": st.column_config.TextColumn("九宫格", width="medium"),
                     "赚钱逻辑": st.column_config.TextColumn("赚钱逻辑", width="large"),
                 }, height=550)

    # ── Zone stock cards ──
    st.markdown('<div class="section-title">目标区域代表股票</div>', unsafe_allow_html=True)
    target_zones = [
        (1, "戴维斯双击", "最理想机会", "↓pe", "↑np"),
        (2, "利润驱动", "赚成长的钱", "→pe", "↑np"),
        (4, "估值修复", "赚修复的钱", "↓pe", "→np"),
    ]
    zcols = st.columns(3)
    for ci, (zid, zname, zdesc, pek, npk) in enumerate(target_zones):
        zone_stocks = [r for r in results if r["zone"] == zid][:3]
        border_color, _ = ZONE_COLORS[zid]
        with zcols[ci]:
            st.markdown(f"""
            <div style="color:{border_color};font-weight:700;font-size:0.95rem;margin-bottom:4px;">{zid}区 · {zname}</div>
            <div style="color:#5a6980;font-size:0.75rem;margin-bottom:8px;">{zdesc}</div>
            """, unsafe_allow_html=True)
            if zone_stocks:
                for s in zone_stocks:
                    st.markdown(f"""
                    <div class="stock-card" style="border-left:3px solid {border_color};">
                        <span class="name">{s['name']}</span><span class="code">{s['code']}</span>
                        <span class="score" style="color:{border_color};float:right;">{s['score_total']:.0f}</span>
                        <div class="meta">PE {s['pe_ttm']:.1f} · 百分位{s.get('pe_quantile_5y',50):.0f}% · 净利增速{s.get('profit_growth',0):+.0f}%</div>
                        <div class="money-type" style="color:{border_color};">→ {s['money_type']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("该区域无股票入选")

    # ── Risk zone warning ──
    st.markdown('<div class="section-title">风险警示区域</div>', unsafe_allow_html=True)
    warn_zones = [(9, "戴维斯双杀"), (6, "情绪下跌"), (7, "价值陷阱")]
    wz_cols = st.columns(3)
    for ci, (zid, zname) in enumerate(warn_zones):
        zone_stocks = [r for r in results if r["zone"] == zid]
        with wz_cols[ci]:
            st.markdown(f'<div style="color:#ef4444;font-weight:700;font-size:0.85rem;">{zid}区 · {zname}</div>', unsafe_allow_html=True)
            if zone_stocks:
                for s in zone_stocks[:3]:
                    st.markdown(f'<div style="font-size:0.75rem;color:#ef4444;padding:3px 0;">{s["name"]} ({s["code"]})</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="font-size:0.75rem;color:#5a6980;">无</div>', unsafe_allow_html=True)

    # ── Download ──
    st.divider()
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button("下载 CSV", csv_buf.getvalue(),
                       file_name=f"pe_np_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                       mime="text/csv")

    # ── Bottom insight ──
    st.markdown("""
    <div style="background:#111827; border:1px solid #1e2a3f; border-radius:10px; padding:18px 24px; margin-top:16px;">
        <div style="font-weight:700; color:#e8edf5; margin-bottom:6px;">买入前问自己一个问题</div>
        <div style="color:#8896b0; font-size:0.85rem;">
            我这笔交易，到底想赚哪一类钱？<br>
            <span style="color:#00d4aa;">赚情绪的钱（PE提升）</span> ·
            <span style="color:#3b82f6;">赚估值修复的钱（PE从低到正常）</span> ·
            <span style="color:#22c55e;">赚企业成长的钱（净利润增长）</span><br>
            不同的答案，对应完全不同的买入、持有、卖出逻辑。<b style="color:#e8edf5;">想清楚了，再下单。</b>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

if run_clicked:
    with st.spinner("正在扫描A股... 财务数据 + PE估值 + 九宫格分类"):
        results, stats = run_screening(max_stocks, top_n, rate_limit, enrich_pe_pct)

    st.session_state.results = results
    st.session_state.stats = stats
    st.session_state.last_run_at = datetime.now().strftime("%H:%M:%S")

    if not results:
        st.warning("没有股票通过筛选。尝试放宽参数。")
        st.stop()

    render_results(results, stats)

elif st.session_state.results is not None:
    st.info(f"上次检索结果 — {st.session_state.last_run_at} | 调整参数后点击按钮开始新一轮检索")
    render_results(st.session_state.results, st.session_state.stats)

else:
    render_welcome()
