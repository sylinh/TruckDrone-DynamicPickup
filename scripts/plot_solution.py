"""
Plot one solution from output file.

Usage:
  python scripts/plot_solution.py --config config.yaml --instance 50.20.4 \
      --solution results/my_benchmark/output50.20.4.txt --sol-index 1
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli import load_yaml_utf8
from src.io_drive import load_instance
from scripts.validate_solution import parse_output_file


def pick_solution(sols: List[dict], idx: int) -> dict:
    if idx is None or idx < 1 or idx > len(sols):
        return sols[0]
    return sols[idx - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--sol-index", type=int, default=1)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    cfg_base = load_yaml_utf8(args.config)
    cfg = dict(cfg_base)
    cfg["instance"] = args.instance
    req_df, _ = load_instance(cfg)
    depot = cfg.get("depot", [0.0, 0.0])

    sols = parse_output_file(Path(args.solution))
    sol = pick_solution(sols, args.sol_index)

    fig, ax = plt.subplots(figsize=(6, 6))
    # plot customers
    c1 = req_df[req_df["drone_ok"] == 0]
    c2 = req_df[req_df["drone_ok"] != 0]
    ax.scatter(c1["x"], c1["y"], c="red", marker="s", label="C1 truck-only")
    ax.scatter(c2["x"], c2["y"], c="blue", marker="o", label="C2 drone-OK")
    ax.scatter([depot[0]], [depot[1]], c="green", marker="*", s=150, label="Depot")

    def plot_routes(veh_routes, style, color):
        for trips in veh_routes.values():
            for trip in trips:
                xs = [depot[0]] + [float(req_df.loc[rid, "x"]) for rid in trip] + [depot[0]]
                ys = [depot[1]] + [float(req_df.loc[rid, "y"]) for rid in trip] + [depot[1]]
                ax.plot(xs, ys, style, color=color, alpha=0.7)

    plot_routes(sol.get("TRUCKS", {}), "-", "tab:orange")
    plot_routes(sol.get("DRONES", {}), "--", "tab:purple")

    ax.set_title(f"Instance {args.instance} - Solution {args.sol_index}")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    outdir = Path(args.outdir) if args.outdir else Path("results/plots")
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{args.instance}_sol{args.sol_index}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
