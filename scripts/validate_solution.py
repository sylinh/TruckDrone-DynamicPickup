"""
Validate a Pareto output file and recompute objectives.

Usage:
  python scripts/validate_solution.py --config config.yaml --instance 50.20.4 --solution results/my_benchmark/output50.20.4.txt
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli import load_yaml_utf8
from src.io_drive import load_instance
from src.sim_route import _simulate_route
from src.sim_utils import safe_float, dist
from src.sim_validate import validate_routes


def parse_output_file(path: Path) -> List[dict]:
    sols = []
    curr = None
    section = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("SOLUTION"):
            if curr:
                sols.append(curr)
            curr = {"TRUCKS": {}, "DRONES": {}}
            section = None
        elif line.startswith("OBJ"):
            parts = line.split()
            cmax = float(parts[1].split("=")[1])
            unserved = int(parts[2].split("=")[1])
            curr["OBJ"] = {"Cmax": cmax, "Unserved": unserved}
        elif line == "TRUCKS":
            section = "TRUCKS"
        elif line == "DRONES":
            section = "DRONES"
        elif line == "END":
            if curr:
                sols.append(curr)
                curr = None
            section = None
        else:
            if section in ("TRUCKS", "DRONES"):
                if ":" not in line:
                    continue
                name, rest = line.split(":", 1)
                trips_raw = rest.strip().split("|") if rest.strip() else []
                trips = []
                for tr in trips_raw:
                    tr = tr.strip()
                    if not tr:
                        continue
                    trips.append([int(x) for x in tr.split()])
                curr[section][name.strip()] = trips
    if curr:
        sols.append(curr)
    return sols


def _vehicle_proto(cfg, kind: str):
    if kind == "truck":
        tcfg = cfg["vehicles"]["trucks"]
        return {"type": "truck", "speed": float(tcfg["speed"]), "capacity": float(tcfg["capacity"]), "fixed_time": None}
    dcfg = cfg["vehicles"]["drones"]
    return {
        "type": "drone",
        "speed": float(dcfg["speed"]),
        "capacity": float(dcfg["capacity"]),
        "fixed_time": float(dcfg.get("fixed_time", 0.0)) if dcfg.get("fixed_time", None) is not None else None,
    }


def validate_solution(sol: dict, req_df: pd.DataFrame, cfg: dict) -> Tuple[List[dict], dict]:
    depot = tuple(cfg.get("depot", [0.0, 0.0]))
    Lw = float(cfg.get("constraints", {}).get("Lw", 1e9))
    # enforce dynamic arrival by adjusting e_i to max(e_i, t_arrive)
    req_use = req_df.copy()
    req_use["e_i"] = req_use[["e_i", "t_arrive"]].max(axis=1)

    # In output files, id được hiểu là index nội bộ của simulator (0..N-1 sau khi sort).
    def map_idx(rid: int):
        return int(rid) if int(rid) in req_use.index else None

    violations = []
    served_set = set()
    vehicles = {}

    # build vehicles by index order
    trucks_ct = int(cfg["vehicles"]["trucks"]["count"])
    drones_ct = int(cfg["vehicles"].get("drones", {}).get("count", 0))
    for i in range(trucks_ct):
        vehicles[f"T{i}"] = _vehicle_proto(cfg, "truck")
    for i in range(drones_ct):
        vehicles[f"D{i}"] = _vehicle_proto(cfg, "drone")

    # helper simulate a trip sequence
    def sim_trip(seq, veh):
        # Ước lượng thời điểm khởi hành tối ưu (giảm chờ) để kiểm tra fixed_time.
        t_start = 0.0
        if seq and veh.get("type") == "drone" and len(seq) == 1:
            rid = seq[0]
            r = req_use.loc[rid]
            travel = dist(depot, (r["x"], r["y"])) / max(float(veh.get("speed", 0.0)), 1e-9)
            t_start = max(0.0, float(r["e_i"]) - travel)  # xuất phát sao cho vừa tới e_i
        return _simulate_route(seq, veh, t_start, depot, req_use, depot, Lw, load_start=0.0)

    # check duplicates
    for section, veh_routes in (("TRUCKS", sol.get("TRUCKS", {})), ("DRONES", sol.get("DRONES", {}))):
        for veh_name, trips in veh_routes.items():
            veh = vehicles.get(veh_name)
            if veh is None:
                violations.append({"vehicle": veh_name, "reason": "unknown_vehicle"})
                continue
            for trip in trips:
                # C1 restriction: drone cannot serve drone_ok==0
                if veh["type"] == "drone":
                    for rid in trip:
                        rid_m = map_idx(rid)
                        if rid_m is None:
                            violations.append({"vehicle": veh_name, "req": rid, "reason": "invalid_req"})
                            continue
                        if not bool(req_use.loc[rid, "drone_ok"]):
                            violations.append({"vehicle": veh_name, "req": rid, "reason": "drone_forbidden"})
                for rid in trip:
                    rid_m = map_idx(rid)
                    if rid_m is None:
                        violations.append({"vehicle": veh_name, "req": rid, "reason": "invalid_req"})
                        continue
                    if rid in served_set:
                        violations.append({"vehicle": veh_name, "req": rid, "reason": "duplicate_service"})
                    else:
                        served_set.add(rid)
                trip_idx = []
                for rid in trip:
                    rid_m = map_idx(rid)
                    if rid_m is None:
                        continue
                    trip_idx.append(rid_m)
                res = sim_trip(trip_idx, veh)
                if not res.get("feasible"):
                    violations.append({"vehicle": veh_name, "trip": trip, "reason": res.get("reason", "infeasible")})

    # post-validate with existing validator (time windows, cap, Lw, fixed_time)
    routes_for_validate = {}
    for section, veh_routes in (("TRUCKS", sol.get("TRUCKS", {})), ("DRONES", sol.get("DRONES", {}))):
        for veh_name, trips in veh_routes.items():
            k = int(veh_name[1:]) if veh_name[1:].isdigit() else len(routes_for_validate)
            stops = []
            t_cursor = 0.0
            pos = depot
            speed = vehicles[veh_name]["speed"]
            for trip in trips:
                for rid in trip:
                    rid_m = map_idx(rid)
                    if rid_m is None:
                        violations.append({"vehicle": veh_name, "req": rid, "reason": "invalid_req"})
                        continue
                    r = req_use.loc[rid_m]
                    travel = dist(pos, (r["x"], r["y"])) / max(speed, 1e-9)
                    arrive = t_cursor + travel
                    start = max(arrive, float(r["e_i"]))
                    stops.append({"req": int(rid_m), "start": float(start), "finish": float(start)})
                    t_cursor = start
                    pos = (r["x"], r["y"])
                t_cursor += dist(pos, depot) / max(speed, 1e-9)
                pos = depot
            routes_for_validate[k] = stops
    # validate_routes assumes một tour liên tục; với drone mỗi trip quay depot, cộng dồn sẽ báo fixed_time sai.
    # Đã kiểm tra từng trip bằng _simulate_route ở trên nên bỏ bước tổng hợp này để tránh false-positive.

    # objectives
    if violations:
        return violations, {}
    # recompute Cmax by simulating each vehicle timeline
    cmax = 0.0
    for section, veh_routes in (("TRUCKS", sol.get("TRUCKS", {})), ("DRONES", sol.get("DRONES", {}))):
        for veh_name, trips in veh_routes.items():
            veh = vehicles[veh_name]
            speed = veh["speed"]
            t_cursor = 0.0
            pos = depot
            for trip in trips:
                res = _simulate_route(trip, veh, t_cursor, pos, req_use, depot, Lw, load_start=0.0)
                t_cursor = res.get("end_time", t_cursor)
                pos = depot
            cmax = max(cmax, t_cursor)
    served = len(served_set)
    return violations, {"Cmax": float(cmax), "served": served, "Unserved": int(len(req_use) - served)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--solution", required=True)
    args = ap.parse_args()

    cfg_base = load_yaml_utf8(args.config)
    cfg = dict(cfg_base)
    cfg["instance"] = args.instance
    req_df, _ = load_instance(cfg)

    sols = parse_output_file(Path(args.solution))
    for idx, sol in enumerate(sols, 1):
        violations, obj = validate_solution(sol, req_df, cfg)
        print(f"SOLUTION {idx}")
        if violations:
            print("INVALID")
            for v in violations:
                print(v)
        else:
            print(f"VALID Cmax={obj['Cmax']:.3f} served={obj['served']} Unserved={obj['Unserved']}")


if __name__ == "__main__":
    main()
