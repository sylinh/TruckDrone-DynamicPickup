from .sim_utils import safe_float, dist


def _simulate_route(seq, veh, t_start, pos_start, req_df, depot, Lw, load_start=None):
    """Simulate a fixed sequence for one vehicle; return feasibility and timing metrics."""
    speed = max(safe_float(veh.get("speed", 0.0), 0.0), 1e-9)
    cap = safe_float(veh.get("capacity", 0.0), 0.0)
    load = safe_float(veh.get("load", 0.0), 0.0) if load_start is None else safe_float(load_start, 0.0)

    t = safe_float(t_start, 0.0)
    pos = pos_start

    pick_times = []
    slack_min = float("inf")
    slack_risk = 0.0

    for idx in seq:
        r = req_df.loc[idx]

        rx = safe_float(r.get("x", None), None)
        ry = safe_float(r.get("y", None), None)
        if rx is None or ry is None:
            return {"feasible": False, "reason": "bad_xy"}

        # Drone constraint
        if veh.get("type") == "drone" and not bool(r.get("drone_ok", 1)):
            return {"feasible": False, "reason": "drone_ok"}

        demand = safe_float(r.get("demand", 0.0), 0.0)
        if load + demand > cap + 1e-9:
            return {"feasible": False, "reason": "capacity"}

        travel = dist(pos, (rx, ry)) / speed
        arrive = t + travel
        e_i = safe_float(r.get("e_i", 0.0), 0.0)
        l_i = safe_float(r.get("l_i", 0.0), 0.0)

        start = max(arrive, e_i)
        if start > l_i + 1e-12:
            return {"feasible": False, "reason": "due"}

        load += demand
        t = start
        pos = (rx, ry)
        pick_times.append(t)

        slack = l_i - t
        slack_min = min(slack_min, slack)
        slack_risk += 0.0 if slack <= 0 else 1.0 / max(slack, 1e-6)

    end_time = t + dist(pos, depot) / speed if seq else safe_float(t_start, 0.0)

    for tp in pick_times:
        if (end_time - tp) > safe_float(Lw, 1e9) + 1e-9:
            return {"feasible": False, "reason": "Lw_wait"}

    if veh.get("type") == "drone" and veh.get("fixed_time", None) is not None:
        if (end_time - safe_float(t_start, 0.0)) > safe_float(veh["fixed_time"], 0.0) + 1e-9:
            return {"feasible": False, "reason": "fixed_time"}

    return {
        "feasible": True,
        "end_time": float(end_time),
        "slack_min": float(slack_min if slack_min != float("inf") else 1e9),
        "slack_risk": float(slack_risk),
        "pick_times": pick_times,
        "end_pos": pos,
    }


def _build_prefix_cache(seq, veh, t_start, pos_start, req_df):
    """Cache prefix times/positions/load to avoid re-simulating from start."""
    times, poses, loads, picks = [], [], [], []
    t = float(t_start)
    pos = pos_start
    load = safe_float(veh.get("load", 0.0), 0.0)
    for idx in seq:
        r = req_df.loc[idx]
        rx = safe_float(r.get("x", None), None)
        ry = safe_float(r.get("y", None), None)
        if rx is None or ry is None:
            break
        speed = max(safe_float(veh.get("speed", 0.0), 0.0), 1e-9)
        travel = dist(pos, (rx, ry)) / speed
        arrive = t + travel
        start = max(arrive, safe_float(r.get("e_i", 0.0), 0.0))
        load += safe_float(r.get("demand", 0.0), 0.0)
        t = start
        pos = (rx, ry)
        times.append(t)
        poses.append(pos)
        loads.append(load)
        picks.append(t)
    return {
        "time_after": times,
        "pos_after": poses,
        "load_after": loads,
        "pick_times": picks,
    }


def _best_insertion_for_vehicle(
    req_idx, veh, queue, t_now, req_df, depot, Lw,
    base_cache=None, diag=None, positions_override=None, due_floor=None, backlog_active=False,
    w_ctx=None, rng=None
):
    """Find best insertion; uses LB prune, cache, lexicographic rank."""
    if rng is None:
        import random
        rng = random.Random()
    t_plan = max(safe_float(t_now, 0.0), safe_float(veh.get("free_at", 0.0), 0.0))
    pos_plan = depot
    load_plan = 0.0

    base_key = (
        veh.get("id", id(veh)),
        veh.get("type"),
        tuple(queue),
        round(t_plan, 3),
        tuple(map(float, pos_plan)),
        float(load_plan),
    )

    if base_cache is not None and base_key not in base_cache:
        if diag is not None:
            diag["cache_misses"] = diag.get("cache_misses", 0) + 1
        base_res = _simulate_route(queue, veh, t_plan, pos_plan, req_df, depot, Lw, load_start=load_plan)
        prefix = _build_prefix_cache(queue, veh, t_plan, pos_plan, req_df)
        base_cache[base_key] = {"base": base_res, "prefix": prefix}
    elif base_cache is not None:
        if diag is not None:
            diag["cache_hits"] = diag.get("cache_hits", 0) + 1
    cached = base_cache[base_key] if base_cache is not None else {
        "base": _simulate_route(queue, veh, t_plan, pos_plan, req_df, depot, Lw, load_start=load_plan),
        "prefix": _build_prefix_cache(queue, veh, t_plan, pos_plan, req_df)
    }
    base = cached.get("base", {})
    prefix_cache = cached.get("prefix", {})
    base_end = base["end_time"] if base.get("feasible") else float("inf")

    best = None
    second = None
    feasible_pos_count = 0
    total_pos = len(queue) + 1

    r_req = req_df.loc[req_idx]
    rx = safe_float(r_req.get("x", None), None)
    ry = safe_float(r_req.get("y", None), None)
    if rx is None or ry is None:
        return None

    demand = safe_float(r_req.get("demand", 0.0), 0.0)
    if demand > safe_float(veh.get("capacity", 0.0), 0.0) + 1e-9:
        return None

    if veh.get("type") == "drone":
        if not bool(r_req.get("drone_ok", 1)):
            return None
        if veh.get("fixed_time", None) is not None:
            speed = max(safe_float(veh.get("speed", 0.0), 0.0), 1e-9)
            t_min = 2.0 * dist(depot, (rx, ry)) / speed
            if t_min > safe_float(veh["fixed_time"], 0.0) + 1e-9:
                return None

    speed_lb = max(safe_float(veh.get("speed", 0.0), 0.0), 1e-9)
    travel_lb = dist(pos_plan, (rx, ry)) / speed_lb
    lb_arrive = t_plan + travel_lb
    e_i = safe_float(r_req.get("e_i", 0.0), 0.0)
    l_i = safe_float(r_req.get("l_i", 0.0), 0.0)
    if max(lb_arrive, e_i) > l_i + 1e-12:
        return None

    due_req = l_i
    urgent = (due_req - safe_float(t_now, 0.0)) < 0.2 * max(safe_float(Lw, 1.0), 1.0)

    n = len(queue)
    if positions_override is not None:
        K = min(n + 1, int(positions_override))
    else:
        K = min(n + 1, 20 if urgent else 15)
    positions = {0, n}
    for i in range(K):
        positions.add(round(i * n / max(K - 1, 1)))
    positions = sorted(positions)

    for pos in positions:
        time_after = prefix_cache.get("time_after", [])
        pos_after = prefix_cache.get("pos_after", [])
        load_after = prefix_cache.get("load_after", [])

        start_t = t_plan if pos == 0 else (time_after[pos - 1] if len(time_after) >= pos else t_plan)
        start_pos = pos_plan if pos == 0 else (pos_after[pos - 1] if len(pos_after) >= pos else pos_plan)
        speed_lb = max(safe_float(veh.get("speed", 0.0), 0.0), 1e-9)
        travel_lb_pos = dist(start_pos, (rx, ry)) / speed_lb
        arrive_lb = start_t + travel_lb_pos
        start_service_lb = max(arrive_lb, e_i)
        finish_lb = start_service_lb + dist((rx, ry), depot) / speed_lb
        if start_service_lb > l_i + 1e-12:
            if diag is not None:
                diag["failed_ins_reason"]["due_lb"] = diag["failed_ins_reason"].get("due_lb", 0) + 1
                if rng.random() < 0.01:
                    res_audit = _simulate_route([req_idx] + list(queue[pos:]), veh, start_t, start_pos, req_df, depot, Lw, load_start=(load_after[pos - 1] if pos > 0 and len(load_after) >= pos else load_plan))
                    if res_audit.get("feasible"):
                        diag["prune_false_negative"] = diag.get("prune_false_negative", 0) + 1
            continue
        if finish_lb - start_service_lb > safe_float(Lw, 1e9) + 1e-9:
            if diag is not None:
                diag["failed_ins_reason"]["Lw_lb"] = diag["failed_ins_reason"].get("Lw_lb", 0) + 1
                if rng.random() < 0.01:
                    res_audit = _simulate_route([req_idx] + list(queue[pos:]), veh, start_t, start_pos, req_df, depot, Lw, load_start=(load_after[pos - 1] if pos > 0 and len(load_after) >= pos else load_plan))
                    if res_audit.get("feasible"):
                        diag["prune_false_negative"] = diag.get("prune_false_negative", 0) + 1
            continue

        seq_tail = [req_idx] + list(queue[pos:])
        start_load = load_plan if pos == 0 else (load_after[pos - 1] if len(load_after) >= pos else load_plan)
        res = _simulate_route(seq_tail, veh, start_t, start_pos, req_df, depot, Lw, load_start=start_load)
        if not res.get("feasible"):
            if diag is not None:
                reason = res.get("reason", "unknown")
                diag["failed_ins_reason"][reason] = diag["failed_ins_reason"].get(reason, 0) + 1
            continue

        prefix_picks = prefix_cache.get("pick_times", [])[:pos]
        all_picks = prefix_picks + (res.get("pick_times") or [])
        end_time = safe_float(res.get("end_time", float("inf")), float("inf"))
        violated = False
        for tp in all_picks:
            if end_time - tp > safe_float(Lw, 1e9) + 1e-9:
                violated = True
                break
        if violated:
            if diag is not None:
                diag["failed_ins_reason"]["Lw_wait"] = diag["failed_ins_reason"].get("Lw_wait", 0) + 1
            continue

        if veh.get("type") == "drone" and veh.get("fixed_time", None) is not None:
            total_duration = end_time - t_plan
            if total_duration > safe_float(veh["fixed_time"], 0.0) + 1e-9:
                if diag is not None:
                    diag["failed_ins_reason"]["fixed_time"] = diag["failed_ins_reason"].get("fixed_time", 0) + 1
                continue

        picks_tail = res.get("pick_times") or []
        t_pick_new = picks_tail[0] if picks_tail else None
        slack_min_res = safe_float(res.get("slack_min", 0.0), 0.0)

        if backlog_active and due_floor is not None:
            slack_margin = 0.01 * max(safe_float(Lw, 1.0), 1.0)
            EPS_local = 1e-6
            if (t_pick_new is None) or (t_pick_new > due_req - EPS_local) or (slack_min_res < slack_margin):
                continue

        feasible_pos_count += 1

        delta_end = end_time - base_end if base_end != float("inf") else end_time - t_plan
        delta_end_norm = float(delta_end) / max(safe_float(Lw, 1.0), 1.0)

        slack_norm = max(0.0, min(1.0, slack_min_res / max(safe_float(Lw, 1.0), 1.0)))

        res_risk_avg = safe_float(res.get("slack_risk", 0.0), 0.0) / max(len(seq_tail), 1)
        base_risk_avg = (safe_float(base.get("slack_risk", 0.0), 0.0) / max(len(queue), 1)) if base.get("feasible") else res_risk_avg
        slack_risk_norm = res_risk_avg / max(base_risk_avg, 1e-6)

        urgency = 1.0 / max(due_req - safe_float(t_now, 0.0), 1.0)
        late_pen = 0.0 if t_pick_new is None else 1.0 / max(due_req - float(t_pick_new), 1.0)

        pressure = (due_req - safe_float(t_now, 0.0)) < 0.2 * max(safe_float(Lw, 1.0), 1.0)
        w_delta_default = 0.4 if pressure else 0.7
        w_late_default = 1.3 if pressure else 0.6
        w_slack_default = -1.5
        w_delta = safe_float((w_ctx or {}).get("w_delta", w_delta_default), w_delta_default)
        w_late = safe_float((w_ctx or {}).get("w_late", w_late_default), w_late_default)
        w_slack = safe_float((w_ctx or {}).get("w_slack", w_slack_default), w_slack_default)

        score = (
            w_delta * delta_end_norm
            + 1.0 * slack_risk_norm
            + 1.2 * urgency
            + w_late * late_pen
            + w_slack * slack_norm
        )

        rank_key = (round(late_pen, 6), round(delta_end_norm, 6), -slack_min_res, round(slack_risk_norm, 6))

        cand = {
            "pos": int(pos),
            "score": float(score),
            "end_time": float(end_time),
            "slack_min": float(res.get("slack_min", 0.0)),
            "slack_risk": float(res.get("slack_risk", 0.0)),
            "delta_end": float(delta_end),
            "due": float(due_req),
            "rank_key": rank_key,
        }

        if (best is None) or (cand["rank_key"] < best.get("rank_key", (float("inf"),))):
            second = best
            best = cand
        elif (second is None) or (cand["rank_key"] < second.get("rank_key", (float("inf"),))):
            second = cand

        if best and (slack_min_res > 0.3 * max(safe_float(Lw, 1.0), 1.0)) and (delta_end_norm < 0.02):
            break

    if best:
        best["feasible_pos_count"] = int(feasible_pos_count)
        best["total_pos"] = int(total_pos)
        if second:
            best["regret_pos"] = safe_float(second.get("score", best["score"]) - best["score"], 0.0)
    return best


def _local_repair(queue, veh, t_now, req_df, depot, Lw, center_pos, radius=1, max_moves=8):
    """Local swap/relocate around insertion position (small radius)."""
    if max_moves <= 0:
        return queue
    best_seq = list(queue)
    if len(best_seq) < 2:
        return best_seq

    t_plan = max(safe_float(t_now, 0.0), safe_float(veh.get("free_at", 0.0), 0.0))
    pos_plan = depot
    load_plan = 0.0
    best_res = _simulate_route(best_seq, veh, t_plan, pos_plan, req_df, depot, Lw, load_start=load_plan)

    n = len(best_seq)
    lo = max(0, int(center_pos) - int(radius))
    hi = min(n - 2, int(center_pos) + int(radius))

    def better(res_a, res_b):
        if bool(res_a.get("feasible")) != bool(res_b.get("feasible")):
            return bool(res_a.get("feasible")) and not bool(res_b.get("feasible"))
        return (-safe_float(res_a.get("slack_min", -1e9), -1e9), safe_float(res_a.get("end_time", 1e9), 1e9)) < (
            -safe_float(res_b.get("slack_min", -1e9), -1e9), safe_float(res_b.get("end_time", 1e9), 1e9)
        )

    moves = 0
    for i in range(lo, hi + 1):
        if moves >= max_moves:
            break
        cand_seq = list(best_seq)
        cand_seq[i], cand_seq[i + 1] = cand_seq[i + 1], cand_seq[i]
        res = _simulate_route(cand_seq, veh, t_plan, pos_plan, req_df, depot, Lw, load_start=load_plan)
        moves += 1
        if better(res, best_res):
            best_seq, best_res = cand_seq, res

    for i in range(lo, min(n, int(center_pos) + int(radius) + 1)):
        for j in range(max(0, i - int(radius)), min(n, i + int(radius) + 1)):
            if i == j:
                continue
            if moves >= max_moves:
                break
            cand_seq = list(best_seq)
            node = cand_seq.pop(i)
            cand_seq.insert(j, node)
            res = _simulate_route(cand_seq, veh, t_plan, pos_plan, req_df, depot, Lw, load_start=load_plan)
            moves += 1
            if better(res, best_res):
                best_seq, best_res = cand_seq, res
        if moves >= max_moves:
            break

    return best_seq
