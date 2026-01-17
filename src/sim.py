import numpy as np
from math import hypot


def dist(a, b):
    return float(hypot(a[0] - b[0], a[1] - b[1]))


def _simulate_route(seq, veh, t_start, pos_start, req_df, depot, Lw):
    """Simulate a fixed pickup order; return feasibility and simple risk metrics."""
    speed = max(veh["speed"], 1e-9)
    cap = veh["capacity"]
    load = veh["load"]
    t = t_start
    pos = pos_start
    pick_times = []
    slack_min = float("inf")
    slack_risk = 0.0

    for idx in seq:
        r = req_df.loc[idx]
        if veh["type"] == "drone" and veh.get("radius", None) is not None:
            if dist(depot, (r["x"], r["y"])) > veh["radius"]:
                return {"feasible": False, "reason": "radius"}
        demand = float(r["demand"])
        if load + demand > cap:
            return {"feasible": False, "reason": "capacity"}
        travel = dist(pos, (r["x"], r["y"])) / speed
        arrive = t + travel
        start = max(arrive, r["e_i"])
        if start > r["l_i"]:
            return {"feasible": False, "reason": "due"}
        load += demand
        t = start  # service time ~0
        pos = (r["x"], r["y"])
        pick_times.append(t)
        slack = r["l_i"] - t
        slack_min = min(slack_min, slack)
        slack_risk += 0.0 if slack <= 0 else 1.0 / max(slack, 1e-6)

    end_time = t + dist(pos, depot) / speed if seq else t_start
    for tp in pick_times:
        if (end_time - tp) > Lw:
            return {"feasible": False, "reason": "Lw"}

    return {
        "feasible": True,
        "end_time": end_time,
        "slack_min": slack_min if slack_min != float("inf") else 1e9,
        "slack_risk": slack_risk,
        "pick_times": pick_times,
    }


def _best_insertion_for_vehicle(req_idx, veh, queue, t_now, req_df, depot, Lw, base_cache=None):
    """Find best insertion position for req_idx in vehicle queue. Returns None if no feasible slot."""
    base_key = (tuple(queue), round(max(t_now, veh["free_at"]), 3))
    if base_cache is not None and base_key not in base_cache:
        base_cache[base_key] = _simulate_route(queue, veh, max(t_now, veh["free_at"]), veh["pos"], req_df, depot, Lw)
    base = base_cache[base_key] if base_cache is not None else _simulate_route(queue, veh, max(t_now, veh["free_at"]), veh["pos"], req_df, depot, Lw)
    base_end = base["end_time"] if base.get("feasible") else float("inf")

    best = None
    feasible_pos_count = 0
    total_pos = len(queue) + 1
    for pos in range(len(queue) + 1):
        seq = list(queue[:pos]) + [req_idx] + list(queue[pos:])
        res = _simulate_route(seq, veh, max(t_now, veh["free_at"]), veh["pos"], req_df, depot, Lw)
        if not res.get("feasible"):
            continue
        feasible_pos_count += 1
        delta_end = res["end_time"] - base_end if base_end != float("inf") else res["end_time"] - max(t_now, veh["free_at"])
        score = delta_end + 0.1 * res.get("slack_risk", 0.0)
        cand = {
            "pos": pos,
            "score": score,
            "end_time": res["end_time"],
            "slack_min": res.get("slack_min", 0.0),
            "slack_risk": res.get("slack_risk", 0.0),
            "delta_end": delta_end,
        }
        if (best is None) or (cand["score"] < best["score"]):
            best = cand
    if best:
        best["feasible_pos_count"] = feasible_pos_count
        best["total_pos"] = total_pos
    return best


def _local_repair(queue, veh, t_now, req_df, depot, Lw, center_pos, radius=1):
    """Try small adjacent swaps around center_pos to rescue deadlines."""
    best_seq = list(queue)
    best_res = _simulate_route(best_seq, veh, max(t_now, veh["free_at"]), veh["pos"], req_df, depot, Lw)
    if not best_res.get("feasible") or len(best_seq) < 2:
        return queue  # keep original; higher layers drop if needed
    n = len(queue)
    lo = max(0, center_pos - radius)
    hi = min(n - 2, center_pos + radius)  # need i+1 valid
    if hi < lo:
        return best_seq
    for i in range(lo, hi + 1):
        cand_seq = list(best_seq)
        cand_seq[i], cand_seq[i + 1] = cand_seq[i + 1], cand_seq[i]
        res = _simulate_route(cand_seq, veh, max(t_now, veh["free_at"]), veh["pos"], req_df, depot, Lw)
        if res.get("feasible") and res["end_time"] < best_res["end_time"]:
            best_seq, best_res = cand_seq, res
    return best_seq


def _mk_vehicles(cfg, depot):
    V = []

    def add(kind, count, speed, capacity, radius=None):
        for _ in range(int(count)):
            V.append({
                "type": kind, "speed": float(speed), "capacity": float(capacity),
                "radius": None if radius is None else float(radius),
                "pos": depot, "load": 0.0, "free_at": 0.0, "queue": []
            })
    tr = cfg["vehicles"]["trucks"]; add("truck", tr["count"], tr["speed"], tr["capacity"])
    dr = cfg["vehicles"].get("drones", {"count": 0})
    if dr["count"] > 0:
        add("drone", dr["count"], dr["speed"], dr["capacity"], dr.get("radius", None))
    return V


def _norm_constants(req_df, cfg, depot):
    D_max = 1.0
    if len(req_df) > 0:
        dists = [dist(depot, (r.x, r.y)) for r in req_df.itertuples()]
        D_max = max(dists + [1.0])
    H = float(cfg.get("horizon", 0.0)) or float(req_df["l_i"].max()) if len(req_df) else 1.0
    H = max(H, 1.0)
    return D_max, H


def run_episode(cfg, policy, req_df):
    """Event-driven simulator with best-insertion, pending pool, and local repair."""
    req_df = req_df.sort_values("t_arrive").reset_index(drop=True)
    N = len(req_df)
    depot = tuple(cfg.get("depot", [0.0, 0.0]))
    vehs = _mk_vehicles(cfg, depot)
    Q_max = max([v["capacity"] for v in vehs] + [1.0])
    D_max, H = _norm_constants(req_df, cfg, depot)
    Lw = float(cfg.get("constraints", {}).get("Lw", 1e9))

    served, dropped = set(), set()
    drop_reasons = []
    timeline = []
    pending = set()
    base_cache = {}
    next_idx = 0
    t = 0.0

    def log_drop(idx, reason, t_now=None, veh_idx=None, detail=None):
        rec = {"req_idx": int(idx), "reason": str(reason), "time": float(t if t_now is None else t_now)}
        if veh_idx is not None:
            rec["vehicle"] = int(veh_idx)
        if detail is not None:
            rec["detail"] = detail
        drop_reasons.append(rec)

    def is_feasible(k, r, load_override=None, t_now=None, pos_override=None):
        V = vehs[k]
        demand = float(r["demand"])
        if demand > V["capacity"]:
            return False
        # disallow drone serving non-dronable requests
        if V["type"] == "drone" and not bool(r.get("drone_ok", 1)):
            return False
        if V["type"] == "drone" and V.get("radius", None) is not None:
            if dist(depot, (r["x"], r["y"])) > V["radius"]:
                return False
        load_now = V["load"] if load_override is None else load_override
        if (load_now + demand) > V["capacity"]:
            return False
        t_ref = t if t_now is None else t_now
        pos_ref = V["pos"] if pos_override is None else pos_override
        travel_to = dist(pos_ref, (r["x"], r["y"])) / max(V["speed"], 1e-9)
        arrive = t_ref + travel_to
        start = max(arrive, r["e_i"])
        if start > r["l_i"]:
            return False
        return True

    def normalize_choice(choice):
        if choice is None:
            return None, None
        if isinstance(choice, dict):
            return choice.get("vehicle"), choice.get("pos")
        if isinstance(choice, (tuple, list)) and len(choice) >= 2:
            return choice[0], choice[1]
        return choice, None

    def assign_request(req_idx, t_now, veh_hint=None, pos_hint=None):
        """Try to insert req_idx. Returns True if queued, False if pushed to pending."""
        best = None
        if veh_hint is not None:
            k = int(veh_hint)
            pos = None
            ins = None
            if pos_hint is not None:
                pos = int(pos_hint)
                seq = list(vehs[k]["queue"][:pos]) + [req_idx] + list(vehs[k]["queue"][pos:])
                res = _simulate_route(seq, vehs[k], max(t_now, vehs[k]["free_at"]), vehs[k]["pos"], req_df, depot, Lw)
                if res.get("feasible"):
                    ins = {"pos": pos, "score": res["end_time"], "end_time": res["end_time"], "slack_min": res.get("slack_min"), "slack_risk": res.get("slack_risk")}
            if ins is None:
                ins = _best_insertion_for_vehicle(req_idx, vehs[k], vehs[k]["queue"], t_now, req_df, depot, Lw, base_cache)
            if ins:
                best = (ins, k)
        if best is None:
            for k, V in enumerate(vehs):
                ins = _best_insertion_for_vehicle(req_idx, V, V["queue"], t_now, req_df, depot, Lw, base_cache)
                if not ins:
                    continue
                cand = (ins["score"], -ins["slack_min"], k, ins)
                if (best is None) or (cand < (best[0]["score"], -best[0]["slack_min"], best[1], best[0])):
                    best = (ins, k)
        if best:
            ins, k = best
            V = vehs[k]
            V["queue"].insert(int(ins["pos"]), req_idx)
            V["queue"] = _local_repair(V["queue"], V, t_now, req_df, depot, Lw, center_pos=int(ins["pos"]), radius=1)
            return True
        pending.add(req_idx)
        return False

    def pull_from_pending(veh_idx, t_now, top_k=5):
        """When vehicle is free, try to pull one pending request into its queue."""
        if not pending:
            return
        V = vehs[veh_idx]
        cand_ids = sorted(list(pending), key=lambda i: req_df.loc[i, "l_i"])[:top_k]
        best = None
        for rid in cand_ids:
            ins = _best_insertion_for_vehicle(rid, V, V["queue"], t_now, req_df, depot, Lw, base_cache)
            if not ins:
                continue
            cand = (ins["score"], -ins["slack_min"], rid, ins)
            if (best is None) or (cand < (best[0]["score"], -best[0]["slack_min"], best[1], best[0])):
                best = (ins, rid)
        if best:
            ins, rid = best
            V["queue"].insert(int(ins["pos"]), rid)
            V["queue"] = _local_repair(V["queue"], V, t_now, req_df, depot, Lw, center_pos=int(ins["pos"]), radius=1)
            pending.discard(rid)

    def expire_list(lst, t_now, reason, veh_idx=None):
        keep = []
        for i in lst:
            if t_now > req_df.loc[i, "l_i"]:
                dropped.add(i)
                log_drop(i, reason, t_now=t_now, veh_idx=veh_idx)
            else:
                keep.append(i)
        return keep

    while True:
        all_done = (len(served) + len(dropped) >= N) and all(len(v["queue"]) == 0 for v in vehs) and (len(pending) == 0)
        if all_done:
            break

        next_arrival = req_df.loc[next_idx, "t_arrive"] if next_idx < N else np.inf
        free_candidates = []
        for v in vehs:
            if v["free_at"] > t or (v.get("queue") and len(v["queue"]) > 0):
                free_candidates.append(v["free_at"])
        next_free = min(free_candidates) if free_candidates else np.inf
        t_next = min(next_arrival, next_free)
        if t_next == np.inf:
            break
        t = max(t, t_next)

        # expire pending + queues at current time
        for rid in list(pending):
            if t > req_df.loc[rid, "l_i"]:
                pending.discard(rid)
                dropped.add(rid)
                log_drop(rid, "pending_expired", t_now=t)
        for k, V in enumerate(vehs):
            V["queue"] = expire_list(V.get("queue", []), t, "expired_waiting_in_queue", veh_idx=k)

        # pending coords for features (arrived, not served/dropped)
        arrived_mask = (req_df["t_arrive"] <= t) & (~req_df.index.isin(served)) & (~req_df.index.isin(dropped))
        pending_coords = list(req_df.loc[arrived_mask, ["x", "y"]].itertuples(index=False, name=None))

        state_common = {
            "time": t, "vehicles": vehs, "req_df": req_df, "depot": depot,
            "D_max": D_max, "H": H, "Q_max": Q_max,
            "pending_coords": pending_coords, "is_feasible": is_feasible,
        }

        # 1) handle new arrivals (best-insertion)
        while next_idx < N and req_df.loc[next_idx, "t_arrive"] <= t:
            state_common["time"] = t
            ins_map = {}
            for k, V in enumerate(vehs):
                info = _best_insertion_for_vehicle(next_idx, V, V.get("queue", []), t, req_df, depot, Lw, base_cache)
                if info:
                    ins_map[k] = info
            state_common["insertion_info"] = ins_map
            choice = policy.route_request(state_common, next_idx)
            veh_hint, pos_hint = normalize_choice(choice)
            assigned = False
            if veh_hint is not None:
                assigned = assign_request(next_idx, t, veh_hint=veh_hint, pos_hint=pos_hint)
            if not assigned:
                assign_request(next_idx, t)
            next_idx += 1

        # 2) free vehicles try to pull from pending
        for k, V in enumerate(vehs):
            if V["free_at"] > t:
                continue
            pull_from_pending(k, t, top_k=5)

        # 3) vehicles that are free serve next customers
        for k, V in enumerate(vehs):
            if V["free_at"] > t:
                continue
            if not V.get("queue"):
                continue
            t_v = max(t, V["free_at"])
            pos_v = V["pos"]
            load_v = V["load"]
            tour = []  # (req_idx, pick_time)
            while V.get("queue"):
                V["queue"] = expire_list(V["queue"], t_v, "expired_waiting_in_queue", veh_idx=k)
                if not V["queue"]:
                    break
                state_common["time"] = t_v
                nxt = policy.select_next(state_common, k)
                if nxt is None:
                    nxt = V["queue"][0]
                if nxt not in V["queue"]:
                    continue
                r = req_df.loc[nxt]
                if nxt in served or nxt in dropped:
                    V["queue"] = [x for x in V["queue"] if x != nxt]
                    continue
                if r["t_arrive"] > t_v:
                    break
                if not is_feasible(k, r, load_override=load_v, t_now=t_v, pos_override=pos_v):
                    V["queue"] = [x for x in V["queue"] if x != nxt]
                    continue

                travel_to = dist(pos_v, (r["x"], r["y"])) / max(V["speed"], 1e-9)
                arrive_cust = t_v + travel_to
                start = max(arrive_cust, r["e_i"])
                if start > r["l_i"]:
                    dropped.add(nxt)
                    log_drop(nxt, "cannot_reach_before_due", t_now=t_v, veh_idx=k)
                    V["queue"] = [x for x in V["queue"] if x != nxt]
                    continue

                t_pick_new = start
                t_depot_new = t_pick_new + dist((r["x"], r["y"]), depot) / max(V["speed"], 1e-9)
                feasible_Lw = True
                for _, t_pick_h in tour:
                    if (t_depot_new - t_pick_h) > Lw:
                        feasible_Lw = False
                        break
                if (t_depot_new - t_pick_new) > Lw:
                    feasible_Lw = False
                if not feasible_Lw:
                    V["queue"] = [x for x in V["queue"] if x != nxt]
                    log_drop(nxt, "violates_Lw", t_now=t_v, veh_idx=k)
                    continue

                t_v = start  # service time ~0
                pos_v = (r["x"], r["y"])
                load_v += float(r["demand"])
                tour.append((nxt, float(t_pick_new)))
                served.add(nxt)
                V["queue"] = [x for x in V["queue"] if x != nxt]
                timeline.append((k, int(nxt), float(start), float(start)))

            travel_back = dist(pos_v, depot) / max(V["speed"], 1e-9)
            V["free_at"] = t_v + travel_back
            V["pos"] = depot
            V["load"] = 0.0

            keep = []
            for i in V.get("queue", []):
                r = req_df.loc[i]
                if V["free_at"] > r["l_i"]:
                    dropped.add(i)
                    log_drop(i, "expired_in_queue_after_tour", t_now=V["free_at"], veh_idx=k)
                else:
                    keep.append(i)
            V["queue"] = keep

    for i in range(N):
        if i not in served and i not in dropped:
            dropped.add(i)
            log_drop(i, "unserved_end_of_sim", t_now=t)

    makespan = max([v["free_at"] for v in vehs] + [0.0])
    stats = {
        "makespan": float(makespan),
        "served": len(served),
        "total": N,
        "dropped": len(dropped),
        "drop_reasons": drop_reasons,
        "drop_breakdown": {r: sum(1 for d in drop_reasons if d["reason"] == r) for r in set(d["reason"] for d in drop_reasons)},
    }
    return stats, timeline
