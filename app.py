#!/usr/bin/env python3
"""A股长期投资选股 — Streamlit Web App.

Usage:
  streamlit run app.py
  # Opens browser at http://localhost:8501
"""

import sys
import os
import time
import io
import json
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Import the screening engine
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "skills", "deep-analysis", "scripts")
sys.path.insert(0, SCRIPTS_DIR)
from screen_stocks import (
    fetch_universe, filter_basic, filter_financial,
    fetch_one_stock, score_stock,
)
from trend_predict import predict_one_stock
from market_regime import get_regime
from cockpit import get_cockpit_data
from dividend_financing_ratio import (
    run_dividend_screening,
)

# ── Page config ──
st.set_page_config(
    page_title="A股量化分析系统",
    page_icon="📊",
    layout="wide",
)

# ── Session state: persist results across reruns ──
SESSION_KEYS = [
    "scr_results", "scr_stats",
    "trd_results", "trd_stats", "trd_portfolio",
    "quant_results", "quant_stats",
    "value_results", "value_stats",
    "dfr_results", "dfr_stats",
    "cockpit_data",
    "last_mode", "last_run_at",
]
for key in SESSION_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None

st.title("A股量化分析系统")
st.caption("九维度基本面 · 趋势预判 · 量化扫描 · 价值投资 · 市场驾驶舱")

# ── History persistence ──
HISTORY_DIR = os.path.join(SCRIPTS_DIR, ".cache", "screen_universe", "history")


def _ensure_history_dir():
    os.makedirs(HISTORY_DIR, exist_ok=True)


def save_history(mode: str, params: dict, stats: dict, results: list, portfolio: list = None):
    """Save a run to history."""
    _ensure_history_dir()
    now = datetime.now()
    prefix_map = {"选股排名": "scr", "趋势预判": "trd", "量化扫描": "qnt", "价值投资": "val", "分红融资比": "dfr"}
    prefix = prefix_map.get(mode, "unk")
    rid = f"{prefix}_{now.strftime('%Y%m%d_%H%M%S')}"
    top3 = []
    for r in results[:3]:
        nm = r.get("name", "?")
        if mode == "选股排名":
            top3.append(f"{nm} {r.get('total', 0):.0f}")
        elif mode == "趋势预判":
            top3.append(f"{nm} 半年{r.get('prob_6m', 0):.1f}%")
        elif mode == "量化扫描":
            top3.append(f"{nm} {r.get('total_score', 0):.1f}")
        else:
            top3.append(f"{nm} {r.get('total', 0):.1f}")
    entry = {"id": rid, "ts": now.isoformat(), "mode": mode, "params": params, "stats": stats, "top3": top3}
    result_path = os.path.join(HISTORY_DIR, f"{rid}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        dump_data = {"id": rid, "ts": now.isoformat(), "mode": mode,
                     "params": params, "stats": stats, "results": results}
        if portfolio:
            dump_data["portfolio"] = portfolio
        json.dump(dump_data, f, ensure_ascii=False, indent=2)
    index_path = os.path.join(HISTORY_DIR, "index.json")
    index = []
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            try:
                index = json.load(f)
            except json.JSONDecodeError:
                index = []
    index.insert(0, entry)
    index = index[:50]
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def load_history_list() -> list:
    """Load history index."""
    index_path = os.path.join(HISTORY_DIR, "index.json")
    if not os.path.exists(index_path):
        return []
    with open(index_path, encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def load_history_detail(rid: str) -> dict:
    """Load full results for a specific run."""
    path = os.path.join(HISTORY_DIR, f"{rid}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def delete_history(rid: str):
    """Delete a history record."""
    idx_path = os.path.join(HISTORY_DIR, "index.json")
    if os.path.exists(idx_path):
        with open(idx_path, encoding="utf-8") as f:
            index = json.load(f)
        index = [e for e in index if e["id"] != rid]
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
    result_path = os.path.join(HISTORY_DIR, f"{rid}.json")
    if os.path.exists(result_path):
        os.remove(result_path)


def clear_all_history():
    """Delete all history records."""
    import shutil
    if os.path.exists(HISTORY_DIR):
        shutil.rmtree(HISTORY_DIR)
        _ensure_history_dir()

mode = st.radio("📋 模式", ["驾驶舱", "选股排名", "趋势预判", "量化扫描", "价值投资", "分红融资比", "历史记录"], horizontal=True)

# ── Sidebar controls ──
with st.sidebar:
    if mode == "历史记录":
        st.header("📜 历史记录")
        st.caption("查看过往的选股和预判结果")

        hist_list = load_history_list()
        if hist_list:
            st.metric("总记录数", f"{len(hist_list)} 条")

        st.divider()

        st.subheader("💡 说明")
        st.markdown("""
        - 每次运行**选股排名**或**趋势预判**后自动保存
        - 点击记录可展开查看完整结果
        - 最多保留 50 条历史记录
        """)

        if hist_list and st.button("🗑️ 清空全部历史", type="secondary", use_container_width=True):
            clear_all_history()
            st.rerun()

    else:
        st.header("⚙️ 参数设置")

        if mode == "驾驶舱":
            st.caption("实时市场全景监控")
            st.divider()
            st.markdown("""
            **数据源说明**
            - 指数快照: baostock
            - 市场宽度: 上证交易所摘要 + 涨停池
            - 风险预警: 涨停板/强势股数据
            - 新闻要闻: 东方财富
            - 北向资金: 沪深港通
            """)
            st.caption("数据缓存 5 分钟，点击按钮强制刷新")
            st.divider()
            st.caption("注: 部分 East Money API 可能因反爬限制不可用")
            max_stocks = 0
            top_n = 0
            rate_limit = 3.0
        else:
            max_options = {
                "快速测试 (50只)": 50,
                "中等 (200只)": 200,
                "标准 (500只)": 500,
                "深度 (1000只)": 1000,
                "全量 (所有A股)": 0,
            }
            max_choice = st.selectbox("扫描范围", list(max_options.keys()), index=2)
            max_stocks = max_options[max_choice]

            top_n = st.slider("输出前 N 名", 10, 50, 30, 5)

            rate_options = {"慢速 (3/s, 更稳)": 3.0, "标准 (5/s)": 5.0, "快速 (8/s)": 8.0, "极速 (15/s)": 15.0}
            default_rate_idx = 1 if mode in ("趋势预判", "量化扫描") else 2
            rate_choice = st.selectbox("查询速度", list(rate_options.keys()), index=default_rate_idx)
            rate_limit = rate_options[rate_choice]

        if mode == "趋势预判":
            sort_options = {"半年以上 ↑": "prob_6m", "3个月 ↑": "prob_3m", "2个月 ↑": "prob_2m", "1个月 ↑": "prob_1m"}
            sort_choice = st.selectbox("排序依据", list(sort_options.keys()), index=0)
            horizon_sort = sort_options[sort_choice]

        if mode == "量化扫描":
            with_news = st.checkbox("抓取新闻舆情", value=True, help="通过东方财富获取近期新闻，做关键词情感分析")
            with_flow = st.checkbox("计算资金流向", value=True, help="获取个股主力资金净流向数据")

        st.divider()

        btn_labels = {
            "驾驶舱": "🔄 刷新数据",
            "选股排名": "🚀 开始选股",
            "趋势预判": "🔮 开始预判",
            "量化扫描": "⚡ 量化扫描",
            "价值投资": "💰 价值排名",
            "分红融资比": "💎 分红融资比选股",
        }
        run_clicked = st.button(btn_labels.get(mode, "▶️ 开始"), type="primary", use_container_width=True)

        st.divider()
        if mode != "驾驶舱":
            with st.expander("筛选标准速查"):
                if mode == "选股排名":
                    st.markdown("""
                    **Tier 1 · 基础过滤**
                    - 排除 ST/\\*ST/N/C/PT
                    - 仅 A 股 (沪深北)

                    **Tier 2 · 财务门槛**
                    - ROE ≥ 8%
                    - 净利 > 0
                    - 营收 > 3亿
                    - 负债率 < 70%

                    **九维度打分 (0-100)**
                    | 维度 | 权重 | 说明 |
                    |---|---|---|
                    | 盈利能力 | 20% | ROE + 利润率(行业Z-score校准) |
                    | 成长性 | 16% | 收入/利润 3年CAGR(行业Z-score校准) |
                    | 财务健康 | 15% | 负债率 + 流动比率 |
                    | 估值 | 13% | PE/PB 分位 |
                    | 护城河 | 8% | 毛利率稳定性 + ROE持续性 |
                    | 政策风口 | 8% | 十五五规划重点领域 |
                    | 机构持仓 | 8% | 国家队/社保/险资/北向/QFII |
                    | 行业周期 | 7% | 周期阶段 + 季节特征 |
                    | 实控风险 | 5% | 实控人性质 + 道德风险 |
                    """)
                elif mode == "趋势预判":
                    st.markdown("""
                    **技术指标 (每周期 0-100)**
                    - 均线排列 (MA5/10/20/60/120)
                    - MACD (金叉/死叉/水上水下)
                    - RSI(14) 强弱
                    - Weinstein 阶段分析
                    - 量价配合
                    - 价格位置

                    **预判周期权重**
                    | 周期 | 技术 | 九维度基本面 |
                    |---|---|---|
                    | 1月 | 70% | 30% |
                    | 2月 | 65% | 35% |
                    | 3月 | 50% | 50% |
                    | 半年+ | 40% | 60% |

                    **增强特性**
                    - 市场牛熊状态显示（沪深300 MA排列，仅供参考）
                    - MACD/RSI 顶底背离检测
                    - 行业内相对排名（Z-score标准化）
                    - RS相对强度 / ADX趋势强度
                    - 自适应概率校准
                    """)
            if mode == "量化扫描":
                st.markdown("""
                **量化三因子 (0-100)**
                - 技术面(40%): 动量/量比/RSI/布林位置
                - 舆情面(30%): 近期新闻关键词情感
                - 资金面(30%): 主力资金净流向

                **预测周期**: 1日 / 3日 / 5日
                """)
            if mode == "价值投资":
                st.markdown("""
                **筛选条件**
                - 股息率 ≥ 4%
                - 央国企（央企/地方国企/国资）
                - ROE ≥ 10%
                - 负债率 < 55%
                - PE/PB 处于历史偏低位置

                **评分维度**
                | 维度 | 权重 |
                |---|---|
                | 股息回报 | 30% |
                | 估值安全边际 | 25% |
                | 基本面质量 | 25% |
                | 稳定性 | 20% |
                """)
            if mode == "分红融资比":
                st.markdown("""
                **金标准（来自知乎 kaer）**
                - 历史累计分红 > 历史累计融资

                **五维评分 (0-100)**
                | 维度 | 权重 | 说明 |
                |---|---|---|
                | 分红融资比 | 35% | 核心金标准，>1才值得投资 |
                | 股息回报 | 25% | 当前股息率+3年均值 |
                | 分红质量 | 20% | 连续分红年数+分红比例合理性 |
                | 基本面 | 10% | ROE+负债率 |
                | 分红承诺 | 10% | 章程是否明确分红比例 |
                """)

        if mode == "分红融资比":
            st.divider()
            st.caption("高级过滤")
            require_ratio = st.checkbox("严格: 仅保留 分红融资比>1", value=True,
                                        help="核心金标准：历史分红必须大于融资")
            require_payout = st.checkbox("分红比例 30%-70%", value=True,
                                         help="过低→抠门，过高→杀鸡取卵")
            exclude_declining = st.checkbox("排除衰退/夕阳行业", value=True,
                                            help="排除钢铁、煤炭、房地产、传统零售等")
            min_yield = st.slider("最低股息率(%)", 0.0, 5.0, 2.0, 0.5,
                                  help="当前股息率低于此值的将被排除")

# ── Cached universe fetch ──
@st.cache_data(ttl=3600, show_spinner=False)
def cached_universe():
    return fetch_universe()

# ── Cached cockpit fetch ──
@st.cache_data(ttl=300, show_spinner="🔄 正在加载市场数据...")
def cached_cockpit_data():
    """Cached cockpit data with 5-minute TTL."""
    return get_cockpit_data()

# ── Main area ──

def color_score(val):
    """Color a score cell based on value."""
    if val >= 8:
        return f"background-color: #d4edda; color: #155724"
    elif val >= 5:
        return f"background-color: #fff3cd; color: #856404"
    else:
        return f"background-color: #f8d7da; color: #721c24"


# ── Plotly chart helpers ──────────────────────────────────────────────────

def _plotly_radar(labels: list, values: list, title: str = "", max_val: float = 10) -> go.Figure:
    """Spider/radar chart for multi-dimension scoring."""
    values_closed = list(values) + [values[0]]
    labels_closed = list(labels) + [labels[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed, theta=labels_closed,
        fill='toself', fillcolor='rgba(88, 166, 255, 0.25)',
        line=dict(color='#58a6ff', width=2.5),
        marker=dict(color='#58a6ff', size=6),
        name=title,
    ))
    fig.update_polar(
        radialaxis=dict(range=[0, max_val], gridcolor='#30363d', tickfont=dict(color='#8b949e', size=10)),
        angularaxis=dict(gridcolor='#30363d', tickfont=dict(color='#e6edf3', size=11)),
        bgcolor='#0d1117',
    )
    fig.update_layout(
        paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
        font=dict(color='#e6edf3'), margin=dict(l=40, r=40, t=40, b=40),
        height=380, showlegend=False,
    )
    return fig


def _plotly_bars(labels: list, values: list, title: str = "", color: str = "#58a6ff",
                 height: int = 300, horizontal: bool = False) -> go.Figure:
    """Styled bar chart matching dark theme."""
    if horizontal:
        fig = go.Figure(go.Bar(
            x=values, y=labels, orientation='h',
            marker=dict(color=color, line=dict(color='#30363d', width=1)),
            text=[f"{v:.1f}" for v in values], textposition='outside',
            textfont=dict(color='#8b949e', size=11),
        ))
    else:
        fig = go.Figure(go.Bar(
            x=labels, y=values,
            marker=dict(color=color, line=dict(color='#30363d', width=1)),
            text=[f"{v:.1f}" for v in values], textposition='outside',
            textfont=dict(color='#8b949e', size=11),
        ))
    fig.update_layout(
        paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
        font=dict(color='#e6edf3'), margin=dict(l=20, r=20, t=20, b=20),
        height=height, showlegend=False,
        xaxis=dict(gridcolor='#21262d', zeroline=False),
        yaxis=dict(gridcolor='#21262d', zeroline=False),
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=14, color='#e6edf3')))
    return fig


def _plotly_grouped_bars(labels: list, series: dict, title: str = "", height: int = 400) -> go.Figure:
    """Grouped bar chart for multi-series comparison (e.g. trend horizons)."""
    colors = ["#58a6ff", "#3fb950", "#d29922", "#f78166"]
    fig = go.Figure()
    for i, (name, vals) in enumerate(series.items()):
        fig.add_trace(go.Bar(
            x=labels, y=vals, name=name,
            marker=dict(color=colors[i % len(colors)], line=dict(color='#30363d', width=1)),
            text=[f"{v:.0f}" for v in vals], textposition='outside',
            textfont=dict(color='#8b949e', size=10),
        ))
    fig.update_layout(
        paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
        font=dict(color='#e6edf3'), margin=dict(l=20, r=20, t=30, b=20),
        height=height, barmode='group',
        legend=dict(font=dict(color='#8b949e'), orientation='h', yanchor='bottom', y=1.02),
        xaxis=dict(gridcolor='#21262d', tickfont=dict(size=10)),
        yaxis=dict(gridcolor='#21262d'),
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=14, color='#e6edf3')))
    return fig


def build_results_df(results):
    """Convert score results to styled DataFrame."""
    rows = []
    for i, r in enumerate(results):
        m = r["metrics"]
        rows.append({
            "#": i + 1,
            "代码": r["code"],
            "名称": r["name"],
            "总分": r["total"],
            "盈利能力": r["profitability"],
            "成长性": r["growth"],
            "财务健康": r["health"],
            "估值": r["valuation"],
            "护城河": r["moat"],
            "政策风口": r["policy"],
            "机构持仓": r["institutional"],
            "行业周期": r.get("cycle", 0),
            "实控风险": r.get("controller", 0),
            "ROE(%)": round(m["roe"], 1),
            "负债率(%)": m["debt"],
            "营收CAGR(%)": m["rev_cagr"],
            "净利CAGR(%)": m["profit_cagr"],
            "政策领域": m.get("policy_name", ""),
            "机构": m.get("inst_summary", ""),
            "周期阶段": m.get("cycle_stage", "—"),
            "风险": ", ".join(m.get("cycle_risk_flags", [])[:2]) if m.get("cycle_risk_flags") else "—",
            "实控人": m.get("ctrl_type", "—"),
            "区域": f"{m.get('ctrl_north_south', '—')}/{m.get('ctrl_region', '—')}",
            "道德风险": m.get("ctrl_moral_risk", "—"),
        })
    return pd.DataFrame(rows)


def run_screening(max_stocks, top_n, rate_limit):
    """Run screening with live progress updates. Returns (results_df, stats)."""
    # Tier 1
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

    status_box.info(f"Tier 1 完成：{len(candidates)} 只进入财务筛选")

    # Progress bar
    prog = st.progress(0, text=f"0 / {len(candidates)}")
    stat_line = st.empty()

    import baostock as bs
    bs.login()
    try:
        results = []
        failures = 0
        for i, (code, name) in enumerate(candidates):
            frac = (i + 1) / len(candidates)
            prog.progress(frac, text=f"{i+1}/{len(candidates)} — {name}")

            time.sleep(1.0 / rate_limit)

            m = fetch_one_stock(code, name)
            if m is None:
                failures += 1
                continue

            fail = filter_financial(m)
            if fail:
                continue

            results.append(score_stock(m))

            if (i + 1) % 25 == 0:
                elapsed = time.monotonic() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                stat_line.text(f"已处理: {i+1} | 通过: {len(results)} | "
                               f"速率: {rate:.1f}/s | 用时: {elapsed:.0f}s")
    finally:
        bs.logout()

    elapsed = time.monotonic() - t0
    prog.empty()
    stat_line.empty()
    status_box.empty()

    results.sort(key=lambda r: r["total"], reverse=True)
    results = results[:top_n]

    stats = {
        "universe": len(df), "t1_passed": len(candidates),
        "t2_passed": len(results), "elapsed": round(elapsed, 1),
        "failures": failures,
    }

    save_history("选股排名",
                 {"max": max_stocks, "top": top_n, "rate": rate_limit},
                 stats, results)
    return build_results_df(results) if results else None, stats


def run_trend_prediction(max_stocks, top_n, rate_limit, horizon_sort):
    """Run trend prediction with live progress. Returns (results, stats)."""
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

    status_box.info(f"Tier 1 完成：{len(candidates)} 只进入多周期趋势+九维度基本面分析（日线/周线/月线）")

    prog = st.progress(0, text=f"0 / {len(candidates)}")
    stat_line = st.empty()

    import baostock as bs
    bs.login()
    regime = get_regime(bs)
    try:
        results = []
        failures = 0
        no_data = 0
        for i, (code, name) in enumerate(candidates):
            frac = (i + 1) / len(candidates)
            prog.progress(frac, text=f"{i+1}/{len(candidates)} — {name}")

            time.sleep(1.0 / rate_limit)

            fin = fetch_one_stock(code, name)
            if fin is None:
                failures += 1
                continue

            fail = filter_financial(fin)
            if fail:
                continue

            pred = predict_one_stock(bs, code, name, fin, regime=regime)
            if pred is None:
                no_data += 1
                continue

            results.append(pred)

            if (i + 1) % 25 == 0:
                elapsed = time.monotonic() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                stat_line.text(f"已处理: {i+1} | 预判: {len(results)} | "
                               f"速率: {rate:.1f}/s | 用时: {elapsed:.0f}s")
    finally:
        bs.logout()

    elapsed = time.monotonic() - t0
    prog.empty()
    stat_line.empty()
    status_box.empty()

    results.sort(key=lambda r: r[horizon_sort], reverse=True)

    # ── Industry-relative momentum adjustment ──
    from industry_benchmark import compute_relative_momentum
    mom_map = compute_relative_momentum(results)
    for r in results:
        mom = mom_map.get(r["code"], {})
        adj = mom.get("combined", 0)
        r["prob_1m"] = round(min(95, max(5, r["prob_1m"] + adj * 1.0)), 1)
        r["prob_2m"] = round(min(95, max(5, r["prob_2m"] + adj * 0.7)), 1)
        r["prob_3m"] = round(min(95, max(5, r["prob_3m"] + adj * 0.3)), 1)
        r["mom_1m_pct"] = mom.get("ret_1m_pct", 50)
        r["mom_3m_pct"] = mom.get("ret_3m_pct", 50)
        r["mom_adj"] = adj
        r["mom_peers"] = mom.get("peer_count", 0)

    # Re-sort after momentum adjustment
    results.sort(key=lambda r: r[horizon_sort], reverse=True)

    # ── Portfolio selection ──
    from trend_predict import select_portfolio
    total_predicted = len(results)
    portfolio = select_portfolio(results)
    results = results[:top_n]

    stats = {
        "universe": len(df), "t1_passed": len(candidates),
        "predicted": total_predicted, "elapsed": round(elapsed, 1),
        "failures": failures, "no_data": no_data,
        "portfolio_n": len(portfolio),
    }

    save_history("趋势预判",
                 {"max": max_stocks, "top": top_n, "rate": rate_limit, "sort": horizon_sort},
                 stats, results, portfolio)
    return results, stats, portfolio


# ── Cockpit Renderer ─────────────────────────────────────────────────────

def _render_cockpit_sections(data: dict):
    """Render all 8 cockpit sections from data dict."""
    fetched_at = data.get("fetched_at", "")

    # ── Section 1: Status bar ──
    idx_data = data.get("index_snapshot", {}).get("indices", [])
    up_count = sum(1 for i in idx_data if i.get("change_pct", 0) > 0)

    if up_count >= 3:
        regime_text = "🟢 强势"
        regime_color = "#26a69a"
    elif up_count >= 2:
        regime_text = "🟡 震荡偏强"
        regime_color = "#d2991d"
    elif up_count >= 1:
        regime_text = "🟠 震荡偏弱"
        regime_color = "#ff9800"
    else:
        regime_text = "🔴 弱势"
        regime_color = "#ef5350"

    st.markdown(f"""
    <div class="cockpit-header" style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <span style="font-size:1.2em; font-weight:700; color:{regime_color};">{regime_text}</span>
            <span style="color:#8b949e; margin-left:16px;">{datetime.now().strftime('%Y-%m-%d')}</span>
        </div>
        <div style="color:#484f58; font-size:0.85em;">
            数据更新于 {fetched_at}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 2: Four index cards ──
    if idx_data:
        icols = st.columns(4)
        for i, idx in enumerate(idx_data):
            with icols[i]:
                err = idx.get("_error")
                pct = idx.get("change_pct", 0)
                delta_color = "normal" if pct >= 0 else "inverse"
                st.metric(
                    label=idx["name"],
                    value=f"{idx['price']:.2f}" if not err else "--",
                    delta=f"{pct:+.2f}%" if not err else None,
                    delta_color=delta_color,
                )
                spark = idx.get("sparkline", [])
                if spark and not err:
                    from cockpit import _sparkline
                    st.markdown(_sparkline(spark), unsafe_allow_html=True)

    st.divider()

    # ── Section 3: Market breadth ──
    breadth = data.get("market_breadth", {})
    bcols = st.columns(6)
    bcols[0].metric("尾盘涨停", breadth.get("limit_up", "--"))
    bcols[1].metric("强势股", breadth.get("advance", "--"))
    bcols[2].metric("成交额", f"{breadth.get('volume_yi', 0):.0f}亿" if breadth.get("volume_yi") else "--")
    sse_pe = breadth.get("sse_pe", 0)
    bcols[3].metric("上证PE", f"{sse_pe:.1f}" if sse_pe else "--")
    sse_to = breadth.get("sse_turnover", 0)
    bcols[4].metric("换手率", f"{sse_to:.2f}%" if sse_to else "--")
    bcols[5].metric("挂牌数", breadth.get("total_stocks", "--"))

    if breadth.get("_error"):
        st.caption(f"全市场宽度不可用，显示上证交易所摘要 | {breadth['_error']}")

    st.divider()


    # ── Section 4: Risk alerts ──
    st.markdown("### ⚠️ 风险预警")
    risk = data.get("risk_alerts", {})
    alerts = risk.get("alerts", [])
    if alerts:
        for a in alerts[:6]:
            sev_color = "#ef5350" if a.get("severity") == "high" else "#ff9800"
            st.markdown(f"""
            <div class="cockpit-card" style="border-left: 3px solid {sev_color}; padding:8px 14px;">
                <b style="color:{sev_color};">[{a.get('type', '')}]</b>
                <span style="color:#e6edf3;">{a.get('description', '')}</span>
            </div>
            """, unsafe_allow_html=True)
    elif risk.get("_error"):
        st.info(f"风险数据暂不可用 — {risk['_error']}")
    else:
        st.success("当前未检测到重大风险信号")

    abnormal = risk.get("abnormal_stocks", [])
    if abnormal:
        adf = pd.DataFrame(abnormal)
        st.dataframe(adf, use_container_width=True, hide_index=True)

    st.divider()

    # ── Section 5: Market news ──
    st.markdown("### 📰 市场要闻")
    news_data = data.get("market_news", {})
    news_list = news_data.get("news", [])
    if news_list:
        ncols = st.columns(2)
        for j, n in enumerate(news_list[:8]):
            with ncols[j % 2]:
                url = n.get('url', '')
                title = n['title'][:80]
                summary = n.get('summary', '')
                time_str = n.get('time', '')
                if url:
                    st.markdown(f"""
                    <div class="news-item">
                        <a href="{url}" target="_blank" style="color:#58a6ff; text-decoration:none; font-weight:600;">
                            {title}
                        </a>
                        <span style="color:#484f58; font-size:0.75em; margin-left:8px;">{time_str[:10]}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="news-item">
                        <span style="color:#e6edf3;">{title}</span>
                        <span style="color:#484f58; font-size:0.75em; margin-left:8px;">{time_str[:10]}</span>
                    </div>
                    """, unsafe_allow_html=True)
    elif news_data.get("_error"):
        st.info(f"新闻数据暂不可用 — {news_data['_error']}")
    else:
        st.caption("暂无新闻数据")

    st.divider()

    # ── Section 6: Northbound flow ──
    st.markdown("### 🌏 北向资金")
    nb = data.get("northbound_flow", {})
    nbcols = st.columns(3)
    nbcols[0].metric(
        "今日净流入",
        f"{nb.get('latest_net_yi', 0):+.1f}亿",
        delta=nb.get("direction", ""),
    )
    nbcols[1].metric("近5日累计", f"{nb.get('cumulative_5d_yi', 0):+.1f}亿")
    nbcols[2].metric("近20日累计", f"{nb.get('cumulative_20d_yi', 0):+.1f}亿")

    daily = nb.get("daily_flows", [])
    if daily:
        dates = [d["date"][-5:] for d in daily[-20:]]
        flows = [d["net_yi"] for d in daily[-20:]]
        bar_colors = ["#3fb950" if v >= 0 else "#f85149" for v in flows]
        fig = go.Figure(go.Bar(
            x=dates, y=flows,
            marker=dict(color=bar_colors, line=dict(color='#30363d', width=1)),
        ))
        fig.update_layout(
            paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
            font=dict(color='#e6edf3'), margin=dict(l=20, r=20, t=10, b=20),
            height=220, showlegend=False,
            xaxis=dict(gridcolor='#21262d', tickfont=dict(size=10)),
            yaxis=dict(gridcolor='#21262d', title='净流入(亿)', titlefont=dict(size=10)),
        )
        st.plotly_chart(fig, use_container_width=True)
    elif nb.get("_error"):
        st.info(f"北向资金数据暂不可用 — {nb['_error']}")

    # Footer
    st.divider()
    st.caption(f"数据获取时间: {fetched_at} · 数据源: baostock + akshare (东方财富) · 仅供参考，不构成投资建议")


def _render_cockpit(run_clicked: bool):
    """Render the 驾驶舱 (market cockpit) page."""
    # ── CSS for dark premium theme ──
    st.markdown("""
    <style>
    .cockpit-card {
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px;
        margin: 8px 0;
        transition: all 0.25s ease;
        position: relative;
        overflow: hidden;
    }
    .cockpit-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #58a6ff, #3fb950, #d29922, #f78166);
        opacity: 0;
        transition: opacity 0.25s ease;
    }
    .cockpit-card:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(88, 166, 255, 0.1);
    }
    .cockpit-card:hover::before {
        opacity: 1;
    }
    .cockpit-header {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        border-bottom: 2px solid #30363d;
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 14px;
    }
    .cockpit-metric-label {
        font-size: 0.8em;
        color: #8b949e;
        letter-spacing: 0.5px;
    }
    .cockpit-metric-value {
        font-size: 1.5em;
        font-weight: 700;
        color: #e6edf3;
    }
    .news-item {
        padding: 10px 14px;
        border-left: 3px solid #30363d;
        margin: 6px 0;
        font-size: 0.85em;
        border-radius: 0 6px 6px 0;
        transition: all 0.2s ease;
        background: #0d1117;
    }
    .news-item:hover {
        border-left-color: #58a6ff;
        background: #161b22;
    }
    .index-card {
        text-align: center;
        padding: 16px 12px;
        border-radius: 12px;
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d;
        transition: all 0.2s ease;
    }
    .index-card:hover {
        border-color: #58a6ff;
        box-shadow: 0 2px 12px rgba(88, 166, 255, 0.08);
    }
    </style>
    """, unsafe_allow_html=True)

    if not run_clicked:
        # Show cached cockpit data if available
        cached = st.session_state.get("cockpit_data")
        if cached is not None and cached.get("fetched_at"):
            data = cached
            fetched_at = data.get("fetched_at", "")
            st.info(f"📋 上次刷新 — {fetched_at} | 点击 🔄 刷新数据 获取最新行情")
            _render_cockpit_sections(data)  # render sections below
            return
        # True welcome screen
        st.markdown("""
        <div style="text-align:center; padding:60px 20px;">
            <div style="font-size:4em; margin-bottom:16px;">🖥️</div>
            <h2 style="color:#e6edf3;">市场驾驶舱</h2>
            <p style="color:#8b949e; font-size:1.1em;">
                实时指数 · 市场宽度 · 风险预警 · 新闻要闻 · 北向资金
            </p>
            <p style="color:#484f58; font-size:0.85em;">
                点击左侧 <b>🔄 刷新数据</b> 开始
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Fetch data (force refresh on button click, use cache otherwise)
    if run_clicked:
        cached_cockpit_data.clear()
    data = cached_cockpit_data()

    if data is None:
        st.error("数据获取失败，请稍后重试")
        return

    st.session_state.cockpit_data = data
    st.session_state.last_mode = mode
    st.session_state.last_run_at = datetime.now().strftime("%H:%M:%S")
    _render_cockpit_sections(data)


# ── Run ──
if mode == "驾驶舱":
    _render_cockpit(run_clicked)
elif mode == "历史记录":
    hist_list = load_history_list()
    if not hist_list:
        st.info("暂无历史记录。运行一次选股或趋势预判后，结果会自动保存到这里。")
    else:
        for i, entry in enumerate(hist_list):
            ts = entry["ts"][:19].replace("T", " ")
            mode_icons = {"选股排名": "🏆", "趋势预判": "🔮", "量化扫描": "⚡", "价值投资": "💰", "分红融资比": "💎"}
            mode_icon = mode_icons.get(entry["mode"], "📊")
            top3_str = " · ".join(entry["top3"]) if entry.get("top3") else "—"

            with st.expander(f"{mode_icon} {ts} | {entry['mode']} | "
                             f"{entry['stats'].get('t2_passed', entry['stats'].get('predicted', '?'))}只 | "
                             f"{entry['stats']['elapsed']}s | {top3_str}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    detail = load_history_detail(entry["id"])
                    if detail and detail.get("results"):
                        if detail["mode"] == "选股排名":
                            df = pd.DataFrame(detail["results"])
                            wanted = ["code","name","total","profitability","growth","health",
                                      "valuation","moat","policy","institutional","cycle","controller"]
                            available = [c for c in wanted if c in df.columns]
                            st.dataframe(df[available],
                                         use_container_width=True, hide_index=True)
                        elif detail["mode"] == "趋势预判":
                            rows = [{"#": j+1, "代码": r["code"], "名称": r["name"],
                                     "1月↑%": r["prob_1m"], "2月↑%": r["prob_2m"],
                                     "3月↑%": r["prob_3m"], "半年↑%": r["prob_6m"],
                                     "盈利": r.get("profitability", 0), "成长": r.get("growth", 0),
                                     "健康": r.get("health", 0), "估值": r.get("valuation", 0),
                                     "护城河": r.get("moat", 0), "政策": r.get("policy", 0),
                                     "机构": r.get("institutional", 0), "周期": r.get("cycle", 0),
                                     "实控": r.get("controller", 0), "总分": r.get("total", 0)}
                                    for j, r in enumerate(detail["results"])]
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                        elif detail["mode"] == "量化扫描":
                            rows = [{"#": j+1, "代码": r["code"], "名称": r["name"],
                                     "综合分": r["total_score"], "1日↑%": r["prob_1d"],
                                     "3日↑%": r["prob_3d"], "5日↑%": r["prob_5d"],
                                     "技术面": r["tech_score"], "舆情面": r["news_score"],
                                     "资金面": r["flow_score"], "动量%": r["roc_5d"],
                                     "量比": r["vol_ratio"], "RSI": r["rsi_7d"],
                                     "舆情": r["news_label"], "资金信号": r["flow_signal"]}
                                    for j, r in enumerate(detail["results"])]
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                        elif detail["mode"] == "价值投资":
                            rows = [{"#": j+1, "代码": r["code"], "名称": r["name"],
                                     "总分": r["total"], "股息率%": r["div_yield"],
                                     "分红年": r["div_years"], "PE分位%": r["pe_q"],
                                     "PB分位%": r["pb_q"], "ROE%": r["roe"],
                                     "负债%": r["debt"], "类型": r["soe"],
                                     "股息得分": r["div_score"], "估值得分": r["val_score"],
                                     "质量得分": r["fund_score"], "稳定得分": r["stab_score"]}
                                    for j, r in enumerate(detail["results"])]
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                        elif detail["mode"] == "分红融资比":
                            rows = [{"#": j+1, "代码": r["code"], "名称": r["name"],
                                     "总分": r["total"], "分红融资比": f"{r['div_fin_ratio']:.1f}x",
                                     "股息率%": r["div_yield"], "连分年": r["consecutive_div_years"],
                                     "分红比例%": r["payout_ratio"], "ROE%": r["roe"],
                                     "负债%": r["debt"], "行业": r.get("industry", "")[:8],
                                     "分红承诺": "✓" if r.get("has_commitment") else "—"}
                                    for j, r in enumerate(detail["results"])]
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                        # Show portfolio if available
                        portfolio = detail.get("portfolio")
                        if portfolio:
                            st.divider()
                            st.subheader(f"💰 推荐买入组合 ({len(portfolio)} 支)")
                            pcols = st.columns(min(4, len(portfolio)))
                            for i, s in enumerate(portfolio):
                                with pcols[i % 4]:
                                    sig = s.get("signal_strength", "—")
                                    border = "2px solid #4caf50" if "强" in sig else ("2px solid #ff9800" if "中等" in sig else "1px solid #999")
                                    st.markdown(f"""<div style="border:{border}; border-radius:10px; padding:10px; margin:5px 0; background:#1a1a2e;"><b style="font-size:1.1em;">#{s['buy_rank']} {s['name']}</b><br><span style="font-size:0.8em; color:#888;">{s['code']}</span><br><span style="color:#4caf50;">半年↑ {s['prob_6m']:.0f}%</span> · <span style="color:#2196f3;">α {s.get('alpha_1m',0):+.1f}%</span><br><span style="font-size:0.8em;">{sig}</span></div>""", unsafe_allow_html=True)
                    else:
                        st.warning("记录文件缺失")

                with col2:
                    st.caption(f"参数: {entry.get('params', {})}")
                    if st.button("🗑️ 删除此记录", key=f"del_{entry['id']}"):
                        delete_history(entry["id"])
                        st.rerun()

elif run_clicked:
    if mode == "选股排名":
        with st.spinner("🚀 正在选股... 扫描财务数据、基本面、估值，预计需要 1-2 分钟"):
            results_df, stats = run_screening(max_stocks, top_n, rate_limit)

        # Save to session state
        st.session_state.scr_results = results_df
        st.session_state.scr_stats = stats
        st.session_state.last_mode = mode
        st.session_state.last_run_at = datetime.now().strftime("%H:%M:%S")

        if results_df is None or results_df.empty:
            st.warning("没有股票通过筛选。尝试放宽参数。")
            st.stop()

        st.toast(f"✅ 选股完成！入围 {len(results_df)} 只，耗时 {stats['elapsed']:.0f}s", icon="✅")

        avg_roe = results_df["ROE(%)"].mean()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏆 入围股票", f"{len(results_df)} 只")
        col2.metric("📈 平均 ROE", f"{avg_roe:.1f}%")
        col3.metric("⚡ 总耗时", f"{stats['elapsed']:.0f}秒")
        col4.metric("📊 通过率", f"{stats['t2_passed']}/{stats['t1_passed']}")

        st.divider()

        col_chart, col_table = st.columns([1, 3])
        with col_chart:
            st.subheader("九维雷达图")
            dims = ["盈利能力", "成长性", "财务健康", "估值", "护城河", "政策风口", "机构持仓", "行业周期", "实控风险"]
            avgs = [
                results_df["盈利能力"].mean(), results_df["成长性"].mean(),
                results_df["财务健康"].mean(), results_df["估值"].mean(),
                results_df["护城河"].mean(), results_df["政策风口"].mean(),
                results_df["机构持仓"].mean(), results_df["行业周期"].mean(),
                results_df["实控风险"].mean(),
            ]
            st.plotly_chart(_plotly_radar(dims, avgs, "入围股票平均分"), use_container_width=True)

        with col_table:
            st.subheader(f"TOP {len(results_df)} 排名")
            styler = results_df.style.applymap(color_score, subset=[
                "盈利能力", "成长性", "财务健康", "估值", "护城河", "政策风口", "机构持仓", "行业周期", "实控风险"
            ]).format({
                "总分": "{:.1f}", "盈利能力": "{:.1f}", "成长性": "{:.1f}",
                "财务健康": "{:.1f}", "估值": "{:.1f}", "护城河": "{:.1f}",
                "政策风口": "{:.1f}", "机构持仓": "{:.1f}", "行业周期": "{:.1f}",
                "实控风险": "{:.1f}",
            })
            st.dataframe(styler, use_container_width=True, hide_index=True,
                         column_config={
                             "#": st.column_config.NumberColumn("#", width="small"),
                             "代码": st.column_config.TextColumn("代码", width="small"),
                             "名称": st.column_config.TextColumn("名称", width="small"),
                             "总分": st.column_config.NumberColumn("总分", width="small"),
                         }, height=600)

        st.divider()
        csv_buf = io.StringIO()
        results_df.to_csv(csv_buf, index=False)
        st.download_button("📥 下载 CSV", csv_buf.getvalue(),
                           file_name=f"stock_screen_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                           mime="text/csv")

    elif mode == "趋势预判":
        with st.spinner("🔮 正在预判趋势... 计算技术指标、市场环境、多周期概率，预计需要 1-2 分钟"):
            results, stats, portfolio = run_trend_prediction(max_stocks, top_n, rate_limit, horizon_sort)

        st.session_state.trd_results = results
        st.session_state.trd_stats = stats
        st.session_state.trd_portfolio = portfolio
        st.session_state.last_mode = mode
        st.session_state.last_run_at = datetime.now().strftime("%H:%M:%S")

        if not results:
            st.warning("没有股票通过预判。尝试放宽参数。")
            st.stop()

        st.toast(f"✅ 预判完成！覆盖 {len(results)} 只，推荐买入 {len(portfolio)} 只", icon="✅")

        # ── 推荐买入组合 (prominent section) ──
        if portfolio:
            st.markdown("---")
            st.subheader(f"💰 推荐买入组合 ({len(portfolio)} 支)")
            st.caption("多信号共振筛选：半年预判 ≥ 50% · 跑赢大盘 · 趋势明确 · 基本面达标 · 等权重持有3-6个月 · 单只止损-8%")

            port_cols = st.columns(len(portfolio) if len(portfolio) <= 4 else 4)
            for i, s in enumerate(portfolio):
                with port_cols[i % 4]:
                    sig = s.get("signal_strength", "—")
                    border = "2px solid #4caf50" if "强" in sig else ("2px solid #ff9800" if "中等" in sig else "1px solid #999")
                    st.markdown(f"""
                    <div style="border:{border}; border-radius:10px; padding:10px; margin:5px 0; background:#1a1a2e;">
                        <b style="font-size:1.1em;">#{s['buy_rank']} {s['name']}</b><br>
                        <span style="font-size:0.8em; color:#888;">{s['code']}</span><br>
                        <span style="color:#4caf50;">半年↑ {s['prob_6m']:.0f}%</span> ·
                        <span style="color:#2196f3;">α {s.get('alpha_1m',0):+.1f}%</span><br>
                        <span style="font-size:0.8em;">ADX {s.get('adx_d',0):.0f} · 总分 {s.get('total',0):.0f}</span><br>
                        <span style="font-size:0.8em;">{sig}</span>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        avg_1m = sum(r["prob_1m"] for r in results) / len(results)
        avg_6m = sum(r["prob_6m"] for r in results) / len(results)
        avg_total = sum(r.get("total", 0) for r in results) / len(results)
        regime_label = results[0].get("regime", "?") if results else "?"
        regime_score = results[0].get("regime_score", 50) if results else 50
        regime_emoji = "🟢" if regime_score >= 65 else ("🟡" if regime_score >= 40 else "🔴")

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("🔮 预判股票", f"{len(results)} 只")
        col2.metric("📈 平均 1月↑", f"{avg_1m:.1f}%")
        col3.metric("🎯 平均 半年↑", f"{avg_6m:.1f}%")
        col4.metric(f"{regime_emoji} 市场状态", f"{regime_label}")
        col5.metric("💎 平均 总分", f"{avg_total:.1f}")
        col6.metric("⚡ 耗时", f"{stats['elapsed']:.0f}s")

        if portfolio:
            st.markdown(f"**组合信号分布:** " + " · ".join(
                f"#{s['buy_rank']} {s['name']} {s.get('signal_strength','')}"
                for s in portfolio[:6]
            ) + (f" ...共{len(portfolio)}支" if len(portfolio) > 6 else ""))

        st.divider()

        # Build DataFrame
        pred_rows = []
        for i, r in enumerate(results):
            pred_rows.append({
                "#": i + 1,
                "代码": r["code"],
                "名称": r["name"],
                "1月↑%": r["prob_1m"],
                "2月↑%": r["prob_2m"],
                "3月↑%": r["prob_3m"],
                "半年↑%": r["prob_6m"],
                "技术D": r["tech_d"],
                "技术W": r["tech_w"],
                "技术M": r["tech_m"],
                "共振": r["mtf_resonance"],
                "阶段": r["stage_d"],
                "背离": f"{'MACD' if r.get('div_macd_d', 0) != 0 else ''}{'↓' if r.get('div_macd_d', 0) == -1 else '↑' if r.get('div_macd_d', 0) == 1 else ''}",
                "盈利": r.get("profitability", 0),
                "成长": r.get("growth", 0),
                "健康": r.get("health", 0),
                "估值": r.get("valuation", 0),
                "护城河": r.get("moat", 0),
                "政策": r.get("policy", 0),
                "机构": r.get("institutional", 0),
                "周期": r.get("cycle", 0),
                "实控": r.get("controller", 0),
                "总分": r.get("total", 0),
            })
        pred_df = pd.DataFrame(pred_rows)

        # Color-code probability columns
        def _prob_bg(val):
            if val >= 70:
                return "background-color: #d4edda; color: #155724"
            elif val >= 50:
                return "background-color: #fff3cd; color: #856404"
            else:
                return "background-color: #f8d7da; color: #721c24"

        styler = pred_df.style.applymap(_prob_bg, subset=["1月↑%", "2月↑%", "3月↑%", "半年↑%"]).applymap(
            color_score, subset=["盈利", "成长", "健康", "估值", "护城河", "政策", "机构", "周期", "实控", "总分"]
        ).format({
            "1月↑%": "{:.1f}", "2月↑%": "{:.1f}", "3月↑%": "{:.1f}", "半年↑%": "{:.1f}",
            "技术D": "{:.1f}", "技术W": "{:.1f}", "技术M": "{:.1f}",
            "盈利": "{:.1f}", "成长": "{:.1f}", "健康": "{:.1f}", "估值": "{:.1f}",
            "护城河": "{:.1f}", "政策": "{:.1f}", "机构": "{:.1f}", "周期": "{:.1f}",
            "实控": "{:.1f}", "总分": "{:.1f}",
        })

        st.subheader(f"TOP {len(results)} 趋势预判排名")
        st.dataframe(styler, use_container_width=True, hide_index=True,
                     column_config={
                         "#": st.column_config.NumberColumn("#", width="small"),
                         "代码": st.column_config.TextColumn("代码", width="small"),
                         "名称": st.column_config.TextColumn("名称", width="small"),
                         "1月↑%": st.column_config.NumberColumn("1月↑%", width="small"),
                         "2月↑%": st.column_config.NumberColumn("2月↑%", width="small"),
                         "3月↑%": st.column_config.NumberColumn("3月↑%", width="small"),
                         "半年↑%": st.column_config.NumberColumn("半年↑%", width="small"),
                         "技术D": st.column_config.NumberColumn("技术D", width="small"),
                         "技术W": st.column_config.NumberColumn("技术W", width="small"),
                         "技术M": st.column_config.NumberColumn("技术M", width="small"),
                         "共振": st.column_config.NumberColumn("共振", width="small"),
                         "阶段": st.column_config.NumberColumn("阶段", width="small"),
                         "总分": st.column_config.NumberColumn("总分", width="small"),
                     }, height=600)

        st.divider()

        # Horizon comparison chart
        st.subheader("各周期预判概率对比")
        top15 = results[:15]
        stock_labels = [f"{r['code']} {r['name']}" for r in top15]
        series = {
            "1个月": [r["prob_1m"] for r in top15],
            "2个月": [r["prob_2m"] for r in top15],
            "3个月": [r["prob_3m"] for r in top15],
            "半年以上": [r["prob_6m"] for r in top15],
        }
        st.plotly_chart(_plotly_grouped_bars(stock_labels, series, height=420), use_container_width=True)

        # CSV download
        csv_buf = io.StringIO()
        pred_df.to_csv(csv_buf, index=False)
        st.download_button("📥 下载 CSV", csv_buf.getvalue(),
                           file_name=f"trend_predict_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                           mime="text/csv")

    elif mode == "量化扫描":
        from quant_mode import run_quant_scan

        with st.spinner("⚡ 正在量化扫描... 获取技术面、舆情面、资金面数据，预计需要 1-2 分钟"):
            results, stats = run_quant_scan(
                max_stocks=max_stocks, top_n=top_n, rate_limit=rate_limit,
                with_news=with_news, with_flow=with_flow,
            )

        st.session_state.quant_results = results
        st.session_state.quant_stats = stats
        st.session_state.last_mode = mode
        st.session_state.last_run_at = datetime.now().strftime("%H:%M:%S")

        save_history("量化扫描",
                     {"max": max_stocks, "top": top_n, "rate": rate_limit,
                      "news": with_news, "flow": with_flow},
                     stats, results)

        if not results:
            st.warning("没有股票通过量化扫描。")
            st.stop()

        st.toast(f"✅ 量化扫描完成！{len(results)} 只股票上榜，耗时 {stats['elapsed']:.0f}s", icon="✅")

        avg_score = sum(r["total_score"] for r in results) / len(results)
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("📈 扫描股票", f"{len(results)} 只")
        col2.metric("🎯 平均综合分", f"{avg_score:.1f}")
        col3.metric("📰 舆情覆盖", f"{stats.get('news_hits', 0)} 只")
        col4.metric("💰 资金覆盖", f"{stats.get('flow_hits', 0)} 只")
        col5.metric("⚡ 耗时", f"{stats['elapsed']:.0f}s")

        st.divider()
        st.subheader(f"⚡ 量化扫描 — 实时涨跌概率 TOP {len(results)}")

        quant_rows = []
        for i, r in enumerate(results):
            quant_rows.append({
                "#": i + 1,
                "代码": r["code"],
                "名称": r["name"],
                "综合分": r["total_score"],
                "1日↑%": r["prob_1d"],
                "3日↑%": r["prob_3d"],
                "5日↑%": r["prob_5d"],
                "技术面": r["tech_score"],
                "舆情面": r["news_score"],
                "资金面": r["flow_score"],
                "5日动量%": r["roc_5d"],
                "量比": r["vol_ratio"],
                "RSI(7)": r["rsi_7d"],
                "舆情": r["news_label"],
                "资金信号": r["flow_signal"],
            })
        quant_df = pd.DataFrame(quant_rows)
        st.dataframe(quant_df, use_container_width=True, hide_index=True,
                     column_config={
                         "#": st.column_config.NumberColumn("#", width="small"),
                         "综合分": st.column_config.NumberColumn("综合分", width="small"),
                         "1日↑%": st.column_config.NumberColumn("1日↑%", width="small"),
                         "3日↑%": st.column_config.NumberColumn("3日↑%", width="small"),
                         "5日↑%": st.column_config.NumberColumn("5日↑%", width="small"),
                     }, height=500)

        # Show news headlines for top 3
        st.divider()
        st.caption("📰 头部股票近期舆情")
        news_cols = st.columns(min(3, len(results)))
        for i, r in enumerate(results[:3]):
            with news_cols[i]:
                headlines = r.get("news_headlines", [])
                if headlines:
                    st.markdown(f"**{r['name']}** ({r['news_label']})")
                    for h in headlines[:5]:
                        icon = "🟢" if h["sentiment"] == "positive" else "🔴"
                        st.markdown(f"{icon} {h['title'][:50]}...")
                else:
                    st.markdown(f"**{r['name']}** — 暂无舆情数据")

    elif mode == "价值投资":
        from value_investing import run_value_screening

        with st.spinner("💰 正在价值筛选... 扫描股息率、央国企、基本面、估值，预计需要 1-2 分钟"):
            results, stats = run_value_screening(
                max_stocks=max_stocks, top_n=top_n, rate_limit=rate_limit,
            )

        st.session_state.value_results = results
        st.session_state.value_stats = stats
        st.session_state.last_mode = mode
        st.session_state.last_run_at = datetime.now().strftime("%H:%M:%S")

        save_history("价值投资",
                     {"max": max_stocks, "top": top_n, "rate": rate_limit},
                     stats, results)

        if not results:
            st.warning("没有股票通过基础筛选。请检查数据源是否正常。")
            st.stop()

        st.toast(f"✅ 价值筛选完成！{len(results)} 只股票入围，耗时 {stats['elapsed']:.0f}s", icon="✅")

        avg_yield = sum(r["div_yield"] for r in results) / len(results)
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("💰 入选股票", f"{len(results)} 只")
        col2.metric("💸 平均股息率", f"{avg_yield:.1f}%")
        col3.metric("🏛️ 央国企占比",
                   f"{sum(1 for r in results if r['is_soe'])}/{len(results)}")
        col4.metric("📊 筛选率",
                   f"{stats['passed']}/{stats['t1_passed']}")
        col5.metric("⚡ 耗时", f"{stats['elapsed']:.0f}s")

        st.divider()
        st.subheader(f"💰 价值投资排名 — 高股息·央国企·基本面好·估值低")

        # Score breakdown chart
        dims = ["股息回报", "估值安全边际", "基本面质量", "稳定性"]
        avgs = [
            sum(r["div_score"] for r in results) / len(results),
            sum(r["val_score"] for r in results) / len(results),
            sum(r["fund_score"] for r in results) / len(results),
            sum(r["stab_score"] for r in results) / len(results),
        ]
        st.plotly_chart(_plotly_bars(dims, avgs, color="#3fb950", height=280), use_container_width=True)

        st.divider()

        value_rows = []
        for i, r in enumerate(results):
            value_rows.append({
                "#": i + 1,
                "代码": r["code"],
                "名称": r["name"],
                "总分": r["total"],
                "股息率%": r["div_yield"],
                "分红年": r["div_years"],
                "PE分位%": r["pe_q"],
                "PB分位%": r["pb_q"],
                "ROE%": r["roe"],
                "负债%": r["debt"],
                "类型": r["soe"],
                "股息得分": r["div_score"],
                "估值得分": r["val_score"],
                "质量得分": r["fund_score"],
                "稳定得分": r["stab_score"],
            })
        value_df = pd.DataFrame(value_rows)
        st.dataframe(value_df, use_container_width=True, hide_index=True,
                     column_config={
                         "#": st.column_config.NumberColumn("#", width="small"),
                         "总分": st.column_config.NumberColumn("总分", width="small"),
                         "股息率%": st.column_config.NumberColumn("股息率%", format="%.1f%%"),
                     }, height=500)

        st.divider()
        st.caption("💡 **投资逻辑**: 高股息提供安全垫，央国企信用背书减少下行风险，估值低位提供向上空间。"
                   "适合作为底仓配置，持有周期 6-12 个月以上。不涨吃股息，涨了赚差价。")

    elif mode == "分红融资比":
        with st.spinner("💎 正在分红融资比筛选... 查询分红历史、计算融资比、多维度打分，预计需要 2-3 分钟"):
            results, stats = run_dividend_screening(
                max_stocks=max_stocks, top_n=top_n, rate_limit=rate_limit,
                require_ratio_gt_1=require_ratio,
                require_payout_30_70=require_payout,
                exclude_declining=exclude_declining,
                min_div_yield=min_yield,
            )

        st.session_state.dfr_results = results
        st.session_state.dfr_stats = stats
        st.session_state.last_mode = mode
        st.session_state.last_run_at = datetime.now().strftime("%H:%M:%S")

        save_history("分红融资比",
                     {"max": max_stocks, "top": top_n, "rate": rate_limit,
                      "ratio_gt_1": require_ratio, "payout_30_70": require_payout,
                      "exclude_declining": exclude_declining, "min_yield": min_yield},
                     stats, results)

        if not results:
            st.warning("没有股票通过分红融资比筛选。尝试放宽过滤条件。")
            st.stop()

        st.toast(f"✅ 分红融资比筛选完成！{len(results)} 只股票入围，耗时 {stats['elapsed']:.0f}s", icon="✅")

        avg_ratio = sum(r["div_fin_ratio"] for r in results) / len(results)
        avg_yield = sum(r["div_yield"] for r in results) / len(results)
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("💎 入选股票", f"{len(results)} 只")
        col2.metric("📊 平均分红融资比", f"{avg_ratio:.1f}x")
        col3.metric("💸 平均股息率", f"{avg_yield:.1f}%")
        col4.metric("🔍 全市场扫描",
                   f"融资比>1: {stats.get('ratio_gt_1_count', '?')} 只")
        col5.metric("⚡ 耗时", f"{stats['elapsed']:.0f}s")

        st.divider()
        st.plotly_chart(_plotly_bars(
            ["分红融资比", "股息回报", "分红质量", "基本面", "分红承诺"],
            [sum(r["div_fin_score"] for r in results) / len(results),
             sum(r["yield_score"] for r in results) / len(results),
             sum(r["quality_score"] for r in results) / len(results),
             sum(r["fund_score"] for r in results) / len(results),
             sum(r["commit_score"] for r in results) / len(results)],
            color="#d2991d", height=280,
        ), use_container_width=True)

        st.divider()
        st.subheader(f"💎 分红融资比排名 — 金标准: 历史分红 > 历史融资")

        dfr_rows = []
        for i, r in enumerate(results):
            dfr_rows.append({
                "#": i + 1,
                "代码": r["code"],
                "名称": r["name"],
                "总分": r["total"],
                "分红融资比": f"{r['div_fin_ratio']:.1f}x",
                "累计分红/股": r["total_div_ps"],
                "股息率%": r["div_yield"],
                "连分年": r["consecutive_div_years"],
                "总分红年": r["total_div_years"],
                "分红比例%": r["payout_ratio"],
                "ROE%": r["roe"],
                "负债%": r["debt"],
                "行业": r["industry"][:8] if r["industry"] else "—",
                "分红承诺": "✓" if r["has_commitment"] else "—",
            })
        dfr_df = pd.DataFrame(dfr_rows)

        def _ratio_color(val):
            try:
                v = float(str(val).replace("x", ""))
                if v >= 2: return "background-color: #d4edda; color: #155724"
                elif v >= 1: return "background-color: #fff3cd; color: #856404"
                else: return "background-color: #f8d7da; color: #721c24"
            except Exception:
                return ""

        styler = dfr_df.style.applymap(_ratio_color, subset=["分红融资比"]).applymap(
            color_score, subset=["总分"]
        ).format({
            "股息率%": "{:.1f}", "分红比例%": "{:.0f}",
            "ROE%": "{:.1f}", "负债%": "{:.1f}",
        })

        st.dataframe(styler, use_container_width=True, hide_index=True,
                     column_config={
                         "#": st.column_config.NumberColumn("#", width="small"),
                         "代码": st.column_config.TextColumn("代码", width="small"),
                         "名称": st.column_config.TextColumn("名称", width="small"),
                         "总分": st.column_config.NumberColumn("总分", width="small"),
                         "分红融资比": st.column_config.TextColumn("分红融资比", width="small"),
                         "累计分红/股": st.column_config.NumberColumn("累计分红/股", width="small"),
                     }, height=600)

        st.divider()

        ratio_gt_1_pct = stats.get("ratio_gt_1_pct", 0)
        st.markdown(f"""
        <div class="cockpit-card" style="border-left: 3px solid #d2991d; padding: 16px;">
            <h4 style="color:#d2991d;">📊 扫描统计</h4>
            <p><b>A股全量:</b> {stats['universe']} →
               <b>T1基础:</b> {stats['t1_passed']} →
               <b>分红融资比>1:</b> {stats.get('ratio_gt_1_count', '?')} 只 ({ratio_gt_1_pct}%) →
               <b>最终入选:</b> {stats['passed']} 只</p>
            <p style="color:#8b949e; font-size:0.85em;">
                核心筛选逻辑来自知乎 kaer：一个企业历史累计分红 > 历史融资额，
                才说明管理层德才兼备，值得散户信任。A股约 800/5000 只满足此条件。
            </p>
        </div>
        """, unsafe_allow_html=True)

        csv_buf = io.StringIO()
        dfr_df.to_csv(csv_buf, index=False)
        st.download_button("📥 下载 CSV", csv_buf.getvalue(),
                           file_name=f"dividend_screen_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                           mime="text/csv")

else:
    # ── No button clicked yet — show cached results or welcome ──
    _cached = None
    _cached_stats = None
    _cached_extra = None

    if mode == "选股排名" and st.session_state.scr_results is not None:
        _cached = st.session_state.scr_results
        _cached_stats = st.session_state.scr_stats
    elif mode == "趋势预判" and st.session_state.trd_results is not None:
        _cached = st.session_state.trd_results
        _cached_stats = st.session_state.trd_stats
        _cached_extra = st.session_state.trd_portfolio
    elif mode == "量化扫描" and st.session_state.quant_results is not None:
        _cached = st.session_state.quant_results
        _cached_stats = st.session_state.quant_stats
    elif mode == "价值投资" and st.session_state.value_results is not None:
        _cached = st.session_state.value_results
        _cached_stats = st.session_state.value_stats
    elif mode == "分红融资比" and st.session_state.dfr_results is not None:
        _cached = st.session_state.dfr_results
        _cached_stats = st.session_state.dfr_stats

    last_at = st.session_state.get("last_run_at", "")
    last_mode = st.session_state.get("last_mode", "")

    if _cached is not None and _cached_stats is not None:
        st.info(f"📋 上次检索结果 — {last_mode} · {last_at} | 调整参数后点击按钮开始新一轮检索")

        if mode == "选股排名":
            results_df = _cached
            stats = _cached_stats
            # Re-render the same display as the run path
            st.toast(f"📋 显示上次检索结果 — 入围 {len(results_df)} 只", icon="📋")
            avg_roe = results_df["ROE(%)"].mean()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🏆 入围股票", f"{len(results_df)} 只")
            col2.metric("📈 平均 ROE", f"{avg_roe:.1f}%")
            col3.metric("⚡ 总耗时", f"{stats['elapsed']:.0f}秒")
            col4.metric("📊 通过率", f"{stats['t2_passed']}/{stats['t1_passed']}")
            st.divider()
            col_chart, col_table = st.columns([1, 3])
            with col_chart:
                st.subheader("九维雷达图")
                dims = ["盈利能力", "成长性", "财务健康", "估值", "护城河", "政策风口", "机构持仓", "行业周期", "实控风险"]
                avgs = [results_df["盈利能力"].mean(), results_df["成长性"].mean(), results_df["财务健康"].mean(), results_df["估值"].mean(), results_df["护城河"].mean(), results_df["政策风口"].mean(), results_df["机构持仓"].mean(), results_df["行业周期"].mean(), results_df["实控风险"].mean()]
                st.plotly_chart(_plotly_radar(dims, avgs, "入围股票平均分"), use_container_width=True)
            with col_table:
                st.subheader(f"TOP {len(results_df)} 排名")
                styler = results_df.style.applymap(color_score, subset=["盈利能力", "成长性", "财务健康", "估值", "护城河", "政策风口", "机构持仓", "行业周期", "实控风险"]).format({"总分": "{:.1f}", "盈利能力": "{:.1f}", "成长性": "{:.1f}", "财务健康": "{:.1f}", "估值": "{:.1f}", "护城河": "{:.1f}", "政策风口": "{:.1f}", "机构持仓": "{:.1f}", "行业周期": "{:.1f}", "实控风险": "{:.1f}"})
                st.dataframe(styler, use_container_width=True, hide_index=True, column_config={"#": st.column_config.NumberColumn("#", width="small"), "代码": st.column_config.TextColumn("代码", width="small"), "名称": st.column_config.TextColumn("名称", width="small"), "总分": st.column_config.NumberColumn("总分", width="small")}, height=600)
            st.divider()
            csv_buf = io.StringIO()
            results_df.to_csv(csv_buf, index=False)
            st.download_button("📥 下载 CSV", csv_buf.getvalue(), file_name=f"stock_screen_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")

        elif mode == "趋势预判":
            results = _cached
            stats = _cached_stats
            portfolio = _cached_extra
            st.toast(f"📋 显示上次检索结果 — {len(results)} 只预判 · 推荐 {len(portfolio) if portfolio else 0} 只", icon="📋")
            # [re-render same trend prediction display]
            if portfolio:
                st.markdown("---")
                st.subheader(f"💰 推荐买入组合 ({len(portfolio)} 支)")
                pcols = st.columns(min(4, len(portfolio)))
                for i, s in enumerate(portfolio):
                    with pcols[i % 4]:
                        sig = s.get("signal_strength", "—")
                        border = "2px solid #4caf50" if "强" in sig else ("2px solid #ff9800" if "中等" in sig else "1px solid #999")
                        sig_icon = "🟢" if "强" in sig else ("🟡" if "中等" in sig else "🟠")
                        st.markdown(f"""<div style="border:{border}; border-radius:10px; padding:10px; margin:5px 0; background:#1a1a2e;"><b style="font-size:1.1em;">#{s['buy_rank']} {s['name']}</b><br><span style="font-size:0.8em; color:#888;">{s['code']}</span><br><span style="color:#4caf50;">半年↑ {s['prob_6m']:.0f}%</span> · <span style="color:#2196f3;">α {s.get('alpha_1m',0):+.1f}%</span><br><span style="font-size:0.8em;">ADX {s.get('adx_d',0):.0f} · 总分 {s.get('total',0):.0f}</span><br><span style="font-size:0.8em;">{sig_icon} {sig}</span></div>""", unsafe_allow_html=True)

        elif mode == "量化扫描":
            results = _cached
            stats = _cached_stats
            st.toast(f"📋 显示上次检索结果 — {len(results)} 只上榜", icon="📋")
            avg_score = sum(r["total_score"] for r in results) / len(results)
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("📈 扫描股票", f"{len(results)} 只")
            col2.metric("🎯 平均综合分", f"{avg_score:.1f}")
            col3.metric("📰 舆情覆盖", f"{stats.get('news_hits', 0)} 只")
            col4.metric("💰 资金覆盖", f"{stats.get('flow_hits', 0)} 只")
            col5.metric("⚡ 耗时", f"{stats['elapsed']:.0f}s")
            st.divider()
            st.subheader(f"⚡ 量化扫描 — 实时涨跌概率 TOP {len(results)}")
            quant_rows = []
            for i, r in enumerate(results):
                quant_rows.append({"#": i+1, "代码": r["code"], "名称": r["name"], "综合分": r["total_score"], "1日↑%": r["prob_1d"], "3日↑%": r["prob_3d"], "5日↑%": r["prob_5d"], "技术面": r["tech_score"], "舆情面": r["news_score"], "资金面": r["flow_score"], "5日动量%": r["roc_5d"], "量比": r["vol_ratio"], "RSI(7)": r["rsi_7d"], "舆情": r["news_label"], "资金信号": r["flow_signal"]})
            quant_df = pd.DataFrame(quant_rows)
            st.dataframe(quant_df, use_container_width=True, hide_index=True, column_config={"#": st.column_config.NumberColumn("#", width="small"), "综合分": st.column_config.NumberColumn("综合分", width="small"), "1日↑%": st.column_config.NumberColumn("1日↑%", width="small"), "3日↑%": st.column_config.NumberColumn("3日↑%", width="small"), "5日↑%": st.column_config.NumberColumn("5日↑%", width="small")}, height=500)

        elif mode == "价值投资":
            results = _cached
            stats = _cached_stats
            st.toast(f"📋 显示上次检索结果 — {len(results)} 只入围", icon="📋")
            avg_yield = sum(r["div_yield"] for r in results) / len(results)
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("💰 入选股票", f"{len(results)} 只")
            col2.metric("💸 平均股息率", f"{avg_yield:.1f}%")
            col3.metric("🏛️ 央国企占比", f"{sum(1 for r in results if r['is_soe'])}/{len(results)}")
            col4.metric("📊 筛选率", f"{stats['passed']}/{stats['t1_passed']}")
            col5.metric("⚡ 耗时", f"{stats['elapsed']:.0f}s")
            st.divider()
            st.subheader(f"💰 价值投资排名 — 高股息·央国企·基本面好·估值低")
            value_rows = []
            for i, r in enumerate(results):
                value_rows.append({"#": i+1, "代码": r["code"], "名称": r["name"], "总分": r["total"], "股息率%": r["div_yield"], "分红年": r["div_years"], "PE分位%": r["pe_q"], "PB分位%": r["pb_q"], "ROE%": r["roe"], "负债%": r["debt"], "类型": r["soe"], "股息得分": r["div_score"], "估值得分": r["val_score"], "质量得分": r["fund_score"], "稳定得分": r["stab_score"]})
            value_df = pd.DataFrame(value_rows)
            st.dataframe(value_df, use_container_width=True, hide_index=True, column_config={"#": st.column_config.NumberColumn("#", width="small"), "总分": st.column_config.NumberColumn("总分", width="small"), "股息率%": st.column_config.NumberColumn("股息率%", format="%.1f%%")}, height=500)
            st.divider()
            st.caption("💡 **投资逻辑**: 高股息提供安全垫，央国企信用背书减少下行风险，估值低位提供向上空间。")

        elif mode == "分红融资比":
            results = _cached
            stats = _cached_stats
            st.toast(f"📋 显示上次检索结果 — {len(results)} 只入围", icon="📋")

            avg_ratio = sum(r["div_fin_ratio"] for r in results) / len(results)
            avg_yield = sum(r["div_yield"] for r in results) / len(results)
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("💎 入选股票", f"{len(results)} 只")
            col2.metric("📊 平均分红融资比", f"{avg_ratio:.1f}x")
            col3.metric("💸 平均股息率", f"{avg_yield:.1f}%")
            col4.metric("🔍 全市场扫描",
                       f"融资比>1: {stats.get('ratio_gt_1_count', '?')} 只")
            col5.metric("⚡ 耗时", f"{stats['elapsed']:.0f}s")
            st.divider()
            st.subheader(f"💎 分红融资比排名 — 金标准: 历史分红 > 历史融资")
            dfr_rows = []
            for i, r in enumerate(results):
                dfr_rows.append({
                    "#": i+1, "代码": r["code"], "名称": r["name"],
                    "总分": r["total"], "分红融资比": f"{r['div_fin_ratio']:.1f}x",
                    "累计分红/股": r["total_div_ps"], "股息率%": r["div_yield"],
                    "连分年": r["consecutive_div_years"], "总分红年": r["total_div_years"],
                    "分红比例%": r["payout_ratio"], "ROE%": r["roe"],
                    "负债%": r["debt"], "行业": r["industry"][:8] if r["industry"] else "—",
                    "分红承诺": "✓" if r["has_commitment"] else "—",
                })
            dfr_df = pd.DataFrame(dfr_rows)
            st.dataframe(dfr_df, use_container_width=True, hide_index=True,
                         column_config={
                             "#": st.column_config.NumberColumn("#", width="small"),
                             "总分": st.column_config.NumberColumn("总分", width="small"),
                         }, height=500)
            st.divider()
            st.markdown(f"""
            **扫描统计:** A股 {stats['universe']} → T1 {stats['t1_passed']} →
            融资比>1: {stats.get('ratio_gt_1_count', '?')} 只 ({stats.get('ratio_gt_1_pct', 0)}%) →
            最终入选 {stats['passed']} 只
            """)

    else:
        # ── True welcome screen (no prior results) ──
        if mode == "选股排名":
            st.markdown("""
            ### 使用方法

            1. 左侧设置参数（或直接用默认值）
            2. 点击 **"开始选股"**
            3. 等待扫描完成（500 只约 10-12 分钟）
            4. 查看排名、图表，下载 CSV

            ### 筛选逻辑

            **Tier 1** 基础过滤 → **Tier 2** 财务门槛(ROE≥8%, 营收>3亿, 负债<70%) → **九维度打分排序**
            """)
        elif mode == "分红融资比":
            st.markdown("""
            ### 核心金标准 · 分红融资比 > 1

            来自知乎 **kaer** 的 "真传一句话"：

            > **一个企业的历史累计分红一定要大于它的历史融资额，才值得投资。**

            ### 五维评分体系

            | 维度 | 权重 | 衡量什么 |
            |------|------|----------|
            | **分红融资比** | 35% | 历史累计分红 ÷ 累计融资，>1 才及格 |
            | **股息回报** | 25% | 近3年平均股息率 + 当前股息率 |
            | **分红质量** | 20% | 连续分红年数 + 分红比例是否在30-70% |
            | **基本面** | 10% | ROE + 负债率 |
            | **分红承诺** | 10% | 是否将分红写入公司章程 |

            ### 使用方法

            1. 左侧设置参数（默认扫 500 只约 2-3 分钟）
            2. 点击 **"分红融资比选股"**
            3. 等待扫描完成
            4. 查看排序、下载 CSV

            ### 为什么不用股息率筛选？

            股息率 = 近一年分红/当前股价。大股东可能为套现搞一次性的高分红，
            可能处于周期景气顶点，可能股价暴跌导致股息率虚高。
            **看历史分红融资比，更能反映公司长期回报股东的能力。**
            """)
        else:
            st.markdown("""
            ### 使用方法

            1. 左侧设置参数（默认 500 只约 15-20 分钟）
            2. 点击 **"开始预判"**
            3. 等待趋势分析完成
            4. 查看多周期概率排名、投资组合推荐

            ### 预判维度

            | 指标 | 权重 |
            |------|------|
            | 多周期均线 | 15% |
            | MACD 能量 | 15% |
            | 量价配合 | 15% |
            | 布林带位置 | 10% |
            | 基本面 | 15% |
            | 波动率 | 10% |
            | 动量 | 10% |
            | ADX 趋势 | 10% |
            """)

