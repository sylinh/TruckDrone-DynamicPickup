# src/features.py
from __future__ import annotations
import math
from typing import Any, Dict, List, Tuple

# Bộ ~10 feature dùng chung cho R và S (khác nhau ở state khi gọi)
FEATURE_ORDER: List[str] = [
    "dist_norm",         # 0: distance v->req / D_max
    "demand_norm",       # 1: demand / Q_max
    "rem_capacity_norm", # 2: (cap-load) / Q_k
    "demand_over_rc",    # 3: demand / max(rem_cap, eps)
    "nearest_next_norm", # 4: min dist req->req' / D_max
    "time_to_ready",     # 5: max(0, e_i - t) / H
    "time_to_due",       # 6: max(0, l_i - t) / H
    "waiting_time",      # 7: max(0, t - t_arrive) / H
    "drone_ok",          # 8: 1 if req can be served by drone
    "is_drone",          # 9: 1 if vehicle is drone
    "ins_delta_end",     # 10: delta end-time of best insertion / H
    "ins_slack_min",     # 11: min slack along best insertion / H
    "ins_feasible_frac", # 12: feasible positions / (|queue|+1)
    "queue_size_frac",   # 13: |queue| / total requests
    "queue_demand_gap",  # 14: (cap - sum(queue demand)) / total demand
    "median_dist_norm",  # 15: dist(median queue coords -> req) / D_max
    "travel_time_norm",  # 16: travel time v->req / H
    "demand_total_frac", # 17: demand / total demand
    "time_to_close_from_free", # 18: max(0, l_i - free_at) / H
    "arrive_frac",            # 19: t_arrive / H
]

def _get(obj: Any, *names: str, default=None):
    """Lấy thuộc tính theo nhiều alias: hỗ trợ dict và object."""
    if obj is None:
        return default
    for n in names:
        if isinstance(obj, dict):
            if n in obj:
                return obj[n]
        else:
            if hasattr(obj, n):
                return getattr(obj, n)
    return default

def _pos_xy(obj: Any) -> Tuple[float, float]:
    """Trả (x,y) từ obj: chấp nhận (x,y) | {'x','y'} | {'pos':(x,y)}."""
    p = _get(obj, "pos", default=None)
    if isinstance(p, (tuple, list)) and len(p) >= 2:
        return float(p[0]), float(p[1])
    x = _get(obj, "x", default=None)
    y = _get(obj, "y", default=None)
    if x is not None and y is not None:
        return float(x), float(y)
    return 0.0, 0.0

def featurize(vehicle: Any = None, req: Any = None, sim: Any = None, **state) -> List[float]:
    """
    Trả về vector feature theo FEATURE_ORDER.
      - featurize(vehicle, req, sim)
      - hoặc featurize(vehicle=..., req=..., sim=...)
      - hoặc featurize(state_dict) với keys 'vehicle','req','sim'
    """
    if vehicle is None and "vehicle" in state:
        vehicle = state["vehicle"]
    if req is None and "req" in state:
        req = state["req"]
    if sim is None and "sim" in state:
        sim = state["sim"]

    now = float(_get(sim, "time", "t", "now", default=0.0))
    total_demand_all = float(_get(sim, "total_demand", default=0.0)) or None
    total_reqs = int(_get(sim, "total_reqs", default=0)) or None

    # norm constants
    D_max = float(_get(sim, "D_max", default=1.0)) or 1.0
    H = float(_get(sim, "H", default=1.0)) or 1.0
    Q_max = float(_get(sim, "Q_max", default=1.0)) or 1.0

    # vehicle params
    cap = float(_get(vehicle, "capacity", "cap", default=1.0))
    load = float(_get(vehicle, "load", default=0.0))
    free_at = float(_get(vehicle, "free_at", default=0.0))
    rem_cap = max(0.0, cap - load)
    is_drone = 1.0 if str(_get(vehicle, "type", default="truck")).lower() == "drone" else 0.0
    vx, vy = _pos_xy(vehicle)
    queue_len = 0
    q_obj = _get(vehicle, "queue", default=[])
    if isinstance(q_obj, (list, tuple)):
        queue_len = len(q_obj)

    # request params
    rx, ry = _pos_xy(req)
    demand = float(_get(req, "demand", default=0.0))
    t_arr = float(_get(req, "t_arrive", "r_i", "r", default=0.0))
    e_i = float(_get(req, "e_i", "open_time", "opentime", default=0.0))
    l_i = float(_get(req, "l_i", "close_time", "closetime", default=e_i))
    drone_ok = float(1.0 if bool(_get(req, "ableServiceByDrone", "dronable", "drone_ok", default=1)) else 0.0)

    # dist + nearest-next
    dist_v_req = math.hypot(vx - rx, vy - ry) / D_max

    pending_coords = _get(sim, "pending_coords", default=None)
    nearest_next = 0.0
    if isinstance(pending_coords, list) and pending_coords:
        dmin = None
        for (px, py) in pending_coords:
            d = math.hypot(rx - px, ry - py)
            if d == 0:
                continue
            dmin = d if dmin is None else min(dmin, d)
        nearest_next = (dmin or 0.0) / D_max

    demand_norm = demand / Q_max
    rem_capacity_norm = rem_cap / max(cap, 1e-9)
    demand_over_rc = demand / max(rem_cap, 1e-9)

    time_to_ready = max(0.0, e_i - now) / H
    time_to_due = max(0.0, l_i - now) / H
    waiting_time = max(0.0, now - t_arr) / H

    # insertion-derived metrics (only set when simulator precomputes them)
    veh_idx = _get(sim, "veh_idx", default=None)
    ins_info = {}
    if veh_idx is not None and isinstance(sim, dict):
        ins_info = (sim.get("insertion_info") or {}).get(veh_idx, {}) if isinstance(sim.get("insertion_info"), dict) else {}
    delta_end_norm = float(ins_info.get("delta_end", 0.0)) / H
    slack_min_norm = float(ins_info.get("slack_min", 0.0)) / H
    feasible_cnt = float(ins_info.get("feasible_pos_count", 0.0))
    total_pos = float(ins_info.get("total_pos", queue_len + 1))
    feasible_frac = feasible_cnt / max(total_pos, 1.0)

    # queue-level metrics relative to total requests/demand
    total_req_count = float(total_reqs) if total_reqs else float(len(_get(sim, "req_df", [])) or 1.0)
    queue_size_frac = float(queue_len) / max(total_req_count, 1.0)
    queue_demand = 0.0
    if isinstance(q_obj, (list, tuple)) and hasattr(sim, "get"):
        try:
            req_df = sim.get("req_df", None)
            if req_df is not None:
                for qi in q_obj:
                    queue_demand += float(req_df.loc[qi, "demand"])
        except Exception:
            queue_demand = 0.0
    total_demand_val = total_demand_all if total_demand_all and total_demand_all > 0 else (queue_demand + demand + 1e-9)
    queue_demand_gap = (cap - queue_demand) / max(total_demand_val, 1e-9)

    # median of queue coords -> req distance
    median_dist = 0.0
    if isinstance(q_obj, (list, tuple)) and q_obj and hasattr(sim, "get"):
        try:
            req_df = sim.get("req_df", None)
            if req_df is not None:
                xs = [req_df.loc[qi, "x"] for qi in q_obj]
                ys = [req_df.loc[qi, "y"] for qi in q_obj]
                mx = sorted(xs)[len(xs)//2]
                my = sorted(ys)[len(ys)//2]
                median_dist = math.hypot(mx - rx, my - ry) / D_max
        except Exception:
            median_dist = 0.0

    # travel time to req normalized by horizon
    travel_time_norm = (math.hypot(vx - rx, vy - ry) / max(_get(vehicle, "speed", 1.0), 1e-9)) / H

    # demand fraction over total demand
    demand_total_frac = demand / max(total_demand_val, 1e-9)
    time_to_close_from_free = max(0.0, l_i - free_at) / H
    arrive_frac = t_arr / H

    vec = [
        float(dist_v_req),
        float(demand_norm),
        float(rem_capacity_norm),
        float(demand_over_rc),
        float(nearest_next),
        float(time_to_ready),
        float(time_to_due),
        float(waiting_time),
        float(drone_ok),
        float(is_drone),
        float(delta_end_norm),
        float(slack_min_norm),
        float(feasible_frac),
        float(queue_size_frac),
        float(queue_demand_gap),
        float(median_dist),
        float(travel_time_norm),
        float(demand_total_frac),
        float(time_to_close_from_free),
        float(arrive_frac),
    ]
    # Optional feature mask: either list of indices to keep, or boolean mask same length
    mask = _get(sim, "feature_mask", default=None)
    if isinstance(mask, (list, tuple)):
        if all(isinstance(m, (int, bool)) for m in mask) and len(mask) == len(vec):
            vec = [v if mask[i] else 0.0 for i, v in enumerate(vec)]
        elif all(isinstance(m, int) for m in mask):
            keep = set(mask)
            vec = [v if i in keep else 0.0 for i, v in enumerate(vec)]
    return vec


def featurize_dict(*args, **kwargs) -> Dict[str, float]:
    v = featurize(*args, **kwargs)
    return {k: v[i] for i, k in enumerate(FEATURE_ORDER)}
