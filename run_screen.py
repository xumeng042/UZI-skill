#!/usr/bin/env python3
"""A股长期投资选股工具 — CLI entry.

Usage:
  python run_screen.py --full              # Full scan
  python run_screen.py --max 500 --top 30  # 500 stocks, top 30
"""
import sys
import os
import argparse

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "skills", "deep-analysis", "scripts")
sys.path.insert(0, SCRIPTS_DIR)
from screen_stocks import run_screen, print_table, print_stats, save_json


def main():
    p = argparse.ArgumentParser(description="A股长期价值投资选股")
    p.add_argument("--full", action="store_true", default=True)
    p.add_argument("--top", type=int, default=30, help="输出前N只")
    p.add_argument("--max", type=int, default=500, help="最多处理股票数(默认500)")
    p.add_argument("--rate", type=float, default=10.0, help="每秒查询数(默认10)")
    p.add_argument("--json", type=str, help="JSON输出路径")
    args = p.parse_args()

    results, stats = run_screen(max_stocks=args.max, top_n=args.top, rate_limit=args.rate)
    print_table(results)
    print_stats(stats)

    json_path = save_json(results, stats, args.json)
    print(f"\nJSON: {json_path}")

    # Also save HTML
    html_path = json_path.replace(".json", ".html")
    _save_html(results, stats, html_path)
    print(f"HTML: {html_path}")


def _save_html(results, stats, path):
    rows = ""
    for i, r in enumerate(results):
        m = r["metrics"]
        rows += (
            f"<tr><td>{i+1}</td><td>{r['code']}</td><td>{r['name']}</td>"
            f"<td><b>{r['total']:.1f}</b></td>"
            f"<td>{r['profitability']:.1f}</td><td>{r['growth']:.1f}</td>"
            f"<td>{r['health']:.1f}</td><td>{r['valuation']:.1f}</td><td>{r['moat']:.1f}</td>"
            f"<td><small>ROE:{m['roe']} PE:{m['pe']} 负债:{m['debt']}% 市值:{m['mcap_yi']}亿</small></td></tr>"
        )

    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8">
<title>长期投资选股排名</title><style>
body{{font-family:'PingFang SC','Microsoft YaHei',sans-serif;margin:20px;background:#f5f5f5}}
h1{{color:#333}}table{{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.1)}}
th{{background:#333;color:#fff;padding:10px 8px;text-align:left}}
td{{padding:8px;border-bottom:1px solid #eee}}tr:hover{{background:#f0f7ff}}
</style></head><body>
<h1>🏆 长期投资选股排名</h1>
<p>全市场 {stats['universe']} 只 → T1: {stats['t1_passed']} → 上榜: {stats['t2_passed']} | 耗时 {stats['elapsed']}s</p>
<table><thead><tr><th>#</th><th>代码</th><th>名称</th><th>总分</th><th>盈利</th><th>成长</th><th>健康</th><th>估值</th><th>护城河</th><th>关键指标</th></tr></thead><tbody>
{rows}</tbody></table></body></html>"""

    with open(path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
