"""
Batch runner to generate Pareto sets and HV table across instances.

Usage:
  python scripts/run_benchmark.py --config config.yaml --runs 10 \
      --instances-dir WithTimeWindows --outdir results/my_name

Outputs:
  - results/<outdir>/table.tsv  (instance, pareto_count_after_10_runs, R1, R2, hypervolume, avg_runtime)
  - results/<outdir>/output<instance>.txt  (Pareto set in agreed text format)
"""

from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

# ensure repo root on path when run as script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli import load_yaml_utf8
from src.io_drive import load_instance
from src.policy import build_baseline
from src.sim import run_episode
from src.gp import train_gphh, build_gp_policy_from_pair


def _routes_from_timeline(timeline) -> Dict[int, List[Dict[str, float]]]:
    """
    Chuyển timeline (serve events) -> routes per vehicle, có chèn mốc về depot để tách tour.
    Giả định: mỗi serve xong, xe quay về depot ngay sau khách (theo mô phỏng), nên chèn marker "DEPOT".
    """
    routes = {}
    for ev in timeline:
        if not isinstance(ev, dict):
            continue
        veh = int(ev.get("vehicle"))
        req_idx = int(ev.get("req"))
        start = float(ev.get("start", ev.get("finish", 0.0)))
        finish = float(ev.get("finish", start))
        routes.setdefault(veh, []).append({"req": req_idx, "start": start, "finish": finish})
        # marker quay về depot ngay sau khách này (để tách tour)
        routes[veh].append({"req": "DEPOT", "start": finish, "finish": finish})
    for k in list(routes.keys()):
        routes[k] = sorted(routes[k], key=lambda r: r["start"])
    return routes


def pareto_filter(sols: List[dict]) -> List[dict]:
    """Keep non-dominated wrt (Cmax, Unserved); both min."""
    out = []
    for s in sols:
        dominated = False
        for t in sols:
            if t is s:
                continue
            if (t["Cmax"] <= s["Cmax"] and t["Unserved"] <= s["Unserved"]) and (
                t["Cmax"] < s["Cmax"] or t["Unserved"] < s["Unserved"]
            ):
                dominated = True
                break
        if not dominated:
            out.append(s)
    return out


def hypervolume_2d(pareto: List[dict], R: Tuple[float, float]) -> float:
    """Assume minimization, R is reference point (worse)."""
    if not pareto:
        return 0.0
    pts = sorted([(p["Cmax"], p["Unserved"]) for p in pareto], key=lambda x: x[0])
    hv = 0.0
    prev_x = R[0]
    for x, y in reversed(pts):
        dx = prev_x - x
        dy = R[1] - y
        if dx > 0 and dy > 0:
            hv += dx * dy
        prev_x = x
    return hv


def write_output_file(pareto: List[dict], out_path: Path, cfg: dict):
    """Serialize Pareto solutions to agreed text format with routes."""
    truck_ct = int(cfg["vehicles"]["trucks"]["count"])
    drone_ct = int(cfg["vehicles"].get("drones", {}).get("count", 0))

    def trips_from_routes(routes: Dict[int, List[Dict[str, float]]]):
        trips = {}
        for veh_idx, stops in routes.items():
            trips_list = []
            current = []
            for s in sorted(stops, key=lambda r: r["start"]):
                rid = s["req"]
                if rid == "DEPOT":
                    if current:
                        trips_list.append(current)
                        current = []
                    continue
                rid_int = int(rid)
                current.append(rid_int)
            if current:
                trips_list.append(current)
            trips[veh_idx] = trips_list
        return trips

    lines = []
    for idx, sol in enumerate(pareto, 1):
        lines.append(f"SOLUTION {idx}")
        lines.append(f"OBJ Cmax={sol['Cmax']:.6f} Unserved={int(sol['Unserved'])}")
        trips = trips_from_routes(sol.get("routes", {}))
        lines.append("TRUCKS")
        for k in range(truck_ct):
            seqs = trips.get(k, [])
            trip_str = " | ".join(" ".join(map(str, t)) for t in seqs) if seqs else ""
            lines.append(f"T{k}: {trip_str}".rstrip())
        lines.append("DRONES")
        for d in range(drone_ct):
            seqs = trips.get(truck_ct + d, [])
            trip_str = " | ".join(" ".join(map(str, t)) for t in seqs) if seqs else ""
            lines.append(f"D{d}: {trip_str}".rstrip())
        lines.append("END")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--mode", choices=["baseline", "gp"], default="baseline", help="baseline (Greedy EDD) hoặc gp (train GP cho từng instance, dùng cho mọi run)")
    ap.add_argument("--instances-dir", default="WithTimeWindows")
    ap.add_argument("--outdir", default="results/my_benchmark")
    ap.add_argument("--glob", default=None, help="Chỉ chạy các instance khớp glob (vd: 12.*)")
    args = ap.parse_args()

    cfg_base = load_yaml_utf8(args.config)
    inst_dir = Path(args.instances_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    table_rows = []

    inst_files = sorted([p for p in inst_dir.iterdir() if p.is_file()])
    seen = set()

    for inst_path in inst_files:
        inst_name = inst_path.name
        for suf in (".json.result.jsonc", ".json.result", ".jsonc", ".json", ".txt"):
            if inst_name.endswith(suf):
                inst_name = inst_name[: -len(suf)]
                break
        if inst_name in seen:
            continue
        if args.glob and not Path(inst_name).match(args.glob):
            continue
        seen.add(inst_name)
        cfg = dict(cfg_base)
        cfg["instance"] = inst_name
        req_df, _ = load_instance(cfg)
        if args.mode == "baseline":
            policy = build_baseline()
        else:
            # train GP once per instance, reuse for runs
            pair = train_gphh(cfg)
            policy = build_gp_policy_from_pair(cfg, pair)
        sols = []
        runtimes = []
        for _ in range(args.runs):
            t0 = time.time()
            stats, timeline = run_episode(cfg, policy, req_df)
            runtimes.append(time.time() - t0)
            routes = _routes_from_timeline(timeline)
            sols.append(
                {
                    "Cmax": float(stats["makespan"]),
                    "Unserved": int(stats["total"] - stats["served"]),
                    "routes": routes,  # contains DEPOT markers to split tours on export
                }
            )
        pareto = pareto_filter(sols)
        max_c = max(s["Cmax"] for s in sols)
        max_u = max(s["Unserved"] for s in sols)
        R = (max_c * 1.05, max_u + 1)
        hv = hypervolume_2d(pareto, R)
        write_output_file(pareto, outdir / f"output{inst_name}.txt", cfg)
        table_rows.append(
            {
                "instance_name": inst_name,
                "pareto_count_after_10_runs": len(pareto),
                "reference_point_R": json.dumps(R),
                "hypervolume_value": hv,
                "avg_runtime_seconds": float(np.mean(runtimes)),
            }
        )

    # save table
    header = ["instance_name", "pareto_count_after_10_runs", "reference_point_R", "hypervolume_value", "avg_runtime_seconds"]
    lines = ["\t".join(header)]
    for r in table_rows:
        lines.append("\t".join(str(r[h]) for h in header))
    (outdir / "table.tsv").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved table: {outdir / 'table.tsv'}")


if __name__ == "__main__":
    main()
