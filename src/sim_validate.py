from .sim_utils import safe_float, dist


# -----------------------------
# Validation
# -----------------------------
def validate_routes(routes: dict, req_df, cfg: dict):
    """
    Kiểm tra lời giải routes có thỏa ràng buộc:
      - capacity
      - drone_ok (drone)
      - time window
      - Lw_wait
      - fixed_time (drone)

    routes: {veh_idx: [ {req, start, finish}, ... ]}

    Trả về: danh sách vi phạm (mỗi phần tử là dict).
    """
    violations = []
    depot = tuple(cfg.get("depot", [0.0, 0.0]))

    trucks_ct = int(cfg["vehicles"]["trucks"]["count"])
    drones = cfg["vehicles"].get("drones", {"count": 0})

    for k, stops in routes.items():
        k_int = int(k)
        is_truck = k_int < trucks_ct

        if is_truck:
            speed = float(cfg["vehicles"]["trucks"]["speed"])
            cap = float(cfg["vehicles"]["trucks"]["capacity"])
            fixed_time = None
            v_type = "truck"
        else:
            speed = float(drones.get("speed", 0.0))
            cap = float(drones.get("capacity", 0.0))
            fixed_time = drones.get("fixed_time", None)
            v_type = "drone"

        pos = depot
        load = 0.0
        t = 0.0
        tour_start = None
        picks = []

        for stop in sorted(stops, key=lambda s: safe_float(s.get("start", 0.0), 0.0)):
            rid = int(stop.get("req", -1))
            start_logged = safe_float(stop.get("start", 0.0), 0.0)

            if rid < 0 or rid >= len(req_df):
                violations.append({"vehicle": k_int, "req": rid, "reason": "invalid_req"})
                continue

            r = req_df.loc[rid]
            rx = safe_float(r.get("x", None), None)
            ry = safe_float(r.get("y", None), None)
            if rx is None or ry is None:
                violations.append({"vehicle": k_int, "req": rid, "reason": "bad_xy"})
                continue

            if v_type == "drone":
                if not bool(r.get("drone_ok", 1)):
                    violations.append({"vehicle": k_int, "req": rid, "reason": "drone_ok"})

            travel = dist(pos, (rx, ry)) / max(float(speed), 1e-9)
            arrive = float(t) + travel
            start = max(arrive, safe_float(r["e_i"], 0.0))

            if abs(start - start_logged) > 1e-6:
                start = start_logged

            if start > safe_float(r["l_i"], 0.0) + 1e-12:
                violations.append({"vehicle": k_int, "req": rid, "reason": "time_window", "detail": {"start": float(start), "l_i": float(r["l_i"])}})

            if (float(load) + safe_float(r["demand"], 0.0)) > float(cap) + 1e-9:
                violations.append({"vehicle": k_int, "req": rid, "reason": "capacity", "detail": {"load": float(load), "demand": float(r["demand"]), "cap": float(cap)}})

            if tour_start is None:
                tour_start = float(t)

            t = float(start)
            pos = (rx, ry)
            load += safe_float(r["demand"], 0.0)
            picks.append((rid, float(t)))

        t_depot = float(t) + dist(pos, depot) / max(float(speed), 1e-9) if picks else float(t)

        if (not is_truck) and fixed_time is not None and tour_start is not None:
            if (t_depot - float(tour_start)) > float(fixed_time) + 1e-9:
                violations.append({"vehicle": k_int, "req": None, "reason": "fixed_time", "detail": {"duration": float(t_depot - float(tour_start)), "limit": float(fixed_time)}})

        Lw = float(cfg.get("constraints", {}).get("Lw", 1e9))
        for rid, t_pick in picks:
            wait = float(t_depot) - float(t_pick)
            if wait > float(Lw) + 1e-9:
                violations.append({"vehicle": k_int, "req": rid, "reason": "Lw_wait", "detail": {"wait": float(wait), "Lw": float(Lw)}})

    return violations
