"""
Aggregate evaluation JSONs in results/ and generate CSV + simple plots.

Usage:
  python scripts/plot_results.py --results-dir results --out-dir results/_plots
Requires: pandas, matplotlib.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt


def _collect_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in root.rglob("*vs_benchmark*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        cmp = obj.get("compare", {})
        ms = cmp.get("makespan") or {}
        sr = cmp.get("service_ratio") or {}
        rows.append({
            "instance": obj.get("instance", p.parent.name),
            "mode": obj.get("mode", p.stem),
            "path": str(p),
            "makespan_ours": ms.get("ours"),
            "makespan_benchmark": ms.get("benchmark"),
            "makespan_abs_diff": ms.get("abs_diff"),
            "makespan_impr_pct": ms.get("improvement_pct"),
            "service_ratio_ours": sr.get("ours"),
            "service_ratio_benchmark": sr.get("benchmark"),
            "service_ratio_abs_diff": sr.get("abs_diff"),
            "service_ratio_impr_pct": sr.get("improvement_pct"),
        })
    return rows


def _plot_makespan(df: pd.DataFrame, out_dir: Path, mode: str):
    if df.empty:
        return
    df = df.sort_values("instance")
    idx = range(len(df))
    width = 0.4
    plt.figure(figsize=(max(6, len(df) * 0.6), 4))
    plt.bar([i - width/2 for i in idx], df["makespan_benchmark"], width=width, label="Benchmark", alpha=0.6)
    plt.bar([i + width/2 for i in idx], df["makespan_ours"], width=width, label="Ours", alpha=0.6)
    plt.xticks(idx, df["instance"], rotation=60)
    plt.ylabel("Makespan")
    plt.title(f"Makespan comparison ({mode})")
    plt.legend()
    plt.tight_layout()
    out = out_dir / f"{mode}_makespan.png"
    plt.savefig(out, dpi=200)
    plt.close()


def _plot_improvement(df: pd.DataFrame, out_dir: Path, mode: str):
    if df.empty or "makespan_impr_pct" not in df:
        return
    df = df.sort_values("instance")
    plt.figure(figsize=(max(6, len(df) * 0.6), 4))
    plt.bar(df["instance"], df["makespan_impr_pct"])
    plt.axhline(0, color="k", linewidth=1)
    plt.xticks(rotation=60)
    plt.ylabel("% improvement vs benchmark (makespan)")
    plt.title(f"Improvement vs benchmark ({mode})")
    plt.tight_layout()
    out = out_dir / f"{mode}_improvement_pct.png"
    plt.savefig(out, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results", help="Root results dir (default: results)")
    ap.add_argument("--out-dir", default="results/_plots", help="Where to save CSV/plots")
    args = ap.parse_args()

    root = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _collect_rows(root)
    if not rows:
        print(f"No benchmark comparison JSONs found under {root}")
        return
    df = pd.DataFrame(rows)
    csv_path = out_dir / "summary_vs_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved summary CSV: {csv_path}")

    for mode, dfm in df.groupby("mode"):
        _plot_makespan(dfm, out_dir, mode)
        _plot_improvement(dfm, out_dir, mode)
        print(f"Saved plots for mode={mode} to {out_dir}")


if __name__ == "__main__":
    main()
