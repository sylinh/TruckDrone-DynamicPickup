"""
Sweep Lw (max waiting) values and plot metrics vs benchmark.

Usage examples (from repo root):
  python scripts/tune_lw.py --config config.yaml --instances 6.5.1 --lw 1e3 1e4 1e5
  python scripts/tune_lw.py --config config.yaml --instances 6.5.1 6.5.2 --lw 1e3 5e3 --out results/_lw

Outputs:
  - CSV summary per run in <out_dir>/lw_scan.csv
  - Plots (makespan vs Lw, service_ratio vs Lw) per instance in <out_dir>/
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.io_drive import load_instance
from src.policy import build_baseline
from src.sim import run_episode


def load_yaml_utf8(path: Path) -> Dict[str, Any]:
    import yaml
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def _service_ratio(stats: Dict[str, Any]) -> float:
    tot = stats.get("total", 0)
    served = stats.get("served", 0)
    if tot:
        return served / tot
    return 0.0


def run_one(cfg_base: Dict[str, Any], instance: str, lw: float):
    cfg = dict(cfg_base)
    cfg["instance"] = instance
    cfg.setdefault("constraints", {})
    cfg["constraints"]["Lw"] = float(lw)
    req_df, bench = load_instance(cfg)
    pol = build_baseline()
    stats, _ = run_episode(cfg, pol, req_df)
    row = {
        "instance": instance,
        "Lw": float(lw),
        "makespan": stats.get("makespan"),
        "served": stats.get("served"),
        "total": stats.get("total"),
        "dropped": stats.get("dropped"),
        "service_ratio": _service_ratio(stats),
        "bench_makespan": None,
        "bench_served": None,
    }
    if isinstance(bench, dict):
        summ = bench.get("summary") or {}
        best = summ.get("gp_best") or summ.get("heuristic") or {}
        row["bench_makespan"] = best.get("makespan")
        row["bench_served"] = best.get("served")
    return row


def plot_instance(df: pd.DataFrame, out_dir: Path, inst: str):
    df = df.sort_values("Lw")
    if df.empty:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(df["Lw"], df["makespan"], marker="o", label="Makespan")
    if "bench_makespan" in df and df["bench_makespan"].notna().any():
        plt.axhline(df["bench_makespan"].dropna().iloc[0], color="r", linestyle="--", label="Benchmark makespan")
    plt.xscale("log")
    plt.xlabel("Lw (log scale)")
    plt.ylabel("Makespan")
    plt.title(f"{inst} - Makespan vs Lw")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{inst}_makespan_vs_Lw.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(df["Lw"], df["service_ratio"], marker="o", label="Service ratio")
    plt.xscale("log")
    plt.xlabel("Lw (log scale)")
    plt.ylabel("Service ratio")
    plt.ylim(0, 1.05)
    plt.title(f"{inst} - Service ratio vs Lw")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{inst}_service_ratio_vs_Lw.png", dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    ap.add_argument("--instances", nargs="+", required=True, help="List of instance ids to scan (e.g., 6.5.1 6.5.2)")
    ap.add_argument("--lw", nargs="+", required=True, help="List of Lw values to test (floats)")
    ap.add_argument("--out", default="results/_lw", help="Output directory for CSV/plots")
    args = ap.parse_args()

    cfg = load_yaml_utf8(Path(args.config))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    lws = [float(x) for x in args.lw]
    rows: List[Dict[str, Any]] = []
    for inst in args.instances:
        for lw in lws:
            row = run_one(cfg, inst, lw)
            rows.append(row)
            print(f"{inst} Lw={lw}: makespan={row['makespan']:.3f}, service_ratio={row['service_ratio']:.3f}")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "lw_scan.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    for inst, dfi in df.groupby("instance"):
        plot_instance(dfi, out_dir, inst)
        print(f"Saved plots for {inst} to {out_dir}")


if __name__ == "__main__":
    main()
