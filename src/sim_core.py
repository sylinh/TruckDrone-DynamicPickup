import random
import numpy as np

from .sim_utils import safe_float, dist
from .sim_route import _simulate_route, _best_insertion_for_vehicle, _local_repair


# -----------------------------
# Vehicles / constants
# -----------------------------
def _mk_vehicles(cfg, depot):
    """Khởi tạo danh sách xe từ cfg; mỗi xe có state: pos/load/free_at/queue."""
    V = []

    def add(kind, count, speed, capacity, fixed_time=None):
        for _ in range(int(count)):
            V.append({
                "type": kind,
                "speed": float(speed),
                "capacity": float(capacity),
                "fixed_time": None if fixed_time is None else float(fixed_time),
                "pos": depot,
                "load": 0.0,
                "free_at": 0.0,
                "queue": [],
            })

    tr = cfg["vehicles"]["trucks"]
    add("truck", tr["count"], tr["speed"], tr["capacity"])

    dr = cfg["vehicles"].get("drones", {"count": 0})
    if int(dr.get("count", 0)) > 0:
        add("drone", dr["count"], dr["speed"], dr["capacity"], dr.get("fixed_time", None))

    for i, v in enumerate(V):
        v["id"] = i

    return V


def _norm_constants(req_df, cfg, depot):
    """Tính hằng số chuẩn hoá: D_max (khoảng cách), H (horizon/thời gian)."""
    D_max = 1.0
    if len(req_df) > 0:
        dists = []
        for r in req_df.itertuples():
            rx = safe_float(getattr(r, "x", None), None)
            ry = safe_float(getattr(r, "y", None), None)
            if rx is None or ry is None:
                continue
            dists.append(dist(depot, (rx, ry)))
        D_max = max(dists + [1.0])

    H = safe_float(cfg.get("horizon", 0.0), 0.0) or (safe_float(req_df["l_i"].max(), 1.0) if len(req_df) else 1.0)
    H = max(H, 1.0)
    return float(D_max), float(H)


# -----------------------------
# Episode simulation
# -----------------------------
def run_episode(cfg, policy, req_df):
    """
    Mô phỏng event-driven: arrival -> assign/pending -> pull -> serve -> expire.

    Policy:
      - policy.route_request(state_common, req_idx) -> (veh_idx, pos) hoặc dict {"vehicle":..,"pos":..}
      - policy.select_next(state_common, veh_idx) -> req_idx
    """
    EPS = 1e-6
    req_df = req_df.sort_values("t_arrive").reset_index(drop=True)
    N = len(req_df)
    seed_val = int(cfg.get("seed", 0))
    random.seed(seed_val)

    depot = tuple(cfg.get("depot", [0.0, 0.0]))
    vehs = _mk_vehicles(cfg, depot)

    Q_max = max([v["capacity"] for v in vehs] + [1.0])
    max_speed_all = max([v["speed"] for v in vehs] + [1e-9])
    D_max, H = _norm_constants(req_df, cfg, depot)
    Lw = float(cfg.get("constraints", {}).get("Lw", 1e9))

    served, dropped = set(), set()
    drop_reasons = []
    timeline = []
    pending = set()

    base_cache = {}
    next_idx = 0
    t = 0.0
    total_demand_all = float(req_df["demand"].sum()) if len(req_df) else 0.0

    diag = {
        "pending_size_over_time": [],
        "min_due_remaining": [],
        "count_no_feasible_insertion_on_arrival": 0,
        "failed_ins_reason": {},
        "pending_snapshot": [],
        "prev_pending_size": 0,
        "last_pending_assign_time": 0.0,
        "opp_calls": 0,
        "opp_success": 0,
        "emergency_calls": 0,
        "emergency_success": 0,
        "emergency_eject_calls": 0,
        "emergency_eject_success": 0,
        "urgent_pull_calls": 0,
        "urgent_pull_success": 0,
        "sim_calls_urgent_total": 0,
        "pull_sim_calls_total": 0,
        "pull_calls": 0,
        "opp_backlog_calls": 0,
        "opp_backlog_success": 0,
        "opp_stall_breaks": 0,
        "stall_breaks": 0,
        "decision_ticks": 0,
        "glob_infeasible_checks": 0,
        "glob_infeasible_drops": 0,
        "glob_infeasible_uncertain": 0,
        # extra debug counters (sau fix ETA)
        "eta_invalid_req_xy": 0,
        "eta_all_failed": 0,
        "eta_partial_failed": 0,
        # new diagnostic
        "prune_false_negative": 0,
        "w_delta_sum": 0.0,
        "w_slack_sum": 0.0,
        "w_weight_count": 0,
    }
    rng_global = random.Random(seed_val)
    last_try_time = {}
    next_decision_time = None
    SIM_BUDGET_PER_TICK = 250  # global cap for simulate/try per decision tick to keep runtime stable
    tick_budget = {"remaining": SIM_BUDGET_PER_TICK}

    def reset_tick_budget():
        tick_budget["remaining"] = SIM_BUDGET_PER_TICK

    def spend_sim_budget(amount=1):
        if tick_budget["remaining"] <= 0:
            return False
        tick_budget["remaining"] = max(0, tick_budget["remaining"] - amount)
        return True

    def weight_ctx():
        pend = len(pending)
        if pend > 15:
            w_delta, w_slack = 0.5, -2.0
        elif pend > 8:
            w_delta, w_slack = 0.6, -1.7
        else:
            w_delta, w_slack = 0.7, -1.5
        w_late = 1.3
        diag["w_delta_sum"] += w_delta
        diag["w_slack_sum"] += w_slack
        diag["w_weight_count"] += 1
        return {"w_delta": w_delta, "w_slack": w_slack, "w_late": w_late}

    def best_insertion_with_budget(*args, **kwargs):
        if not spend_sim_budget():
            return None
        kwargs.setdefault("w_ctx", weight_ctx())
        kwargs.setdefault("rng", rng_global)
        return _best_insertion_for_vehicle(*args, **kwargs)

    def log_drop(idx, reason, t_now=None, veh_idx=None, detail=None):
        rec = {"req_idx": int(idx), "reason": str(reason), "time": float(t if t_now is None else t_now)}
        if veh_idx is not None:
            rec["vehicle"] = int(veh_idx)
        if detail is not None:
            rec["detail"] = detail
        drop_reasons.append(rec)

    def is_feasible(k, r, load_override=None, t_now=None, pos_override=None):
        V = vehs[int(k)]
        demand = safe_float(r.get("demand", 0.0), 0.0)

        if demand > safe_float(V["capacity"], 0.0) + 1e-9:
            return False

        # Drone constraints
        if V["type"] == "drone":
            if not bool(r.get("drone_ok", 1)):
                return False
            # Quick fixed_time filter: depot->req->depot
            if V.get("fixed_time", None) is not None:
                rx = safe_float(r.get("x", None), None)
                ry = safe_float(r.get("y", None), None)
                if rx is None or ry is None:
                    return False
                speed = max(safe_float(V.get("speed", 0.0), 0.0), 1e-9)
                t_min = 2.0 * dist(depot, (rx, ry)) / speed
                if t_min > safe_float(V["fixed_time"], 0.0) + 1e-9:
                    return False

        load_now = safe_float(V["load"], 0.0) if load_override is None else safe_float(load_override, 0.0)
        if (load_now + demand) > safe_float(V["capacity"], 0.0) + 1e-9:
            return False

        t_ref = safe_float(t, 0.0) if t_now is None else safe_float(t_now, 0.0)
        pos_ref = V["pos"] if pos_override is None else pos_override
        px = safe_float(pos_ref[0], None) if isinstance(pos_ref, (tuple, list)) and len(pos_ref) >= 2 else None
        py = safe_float(pos_ref[1], None) if isinstance(pos_ref, (tuple, list)) and len(pos_ref) >= 2 else None
        if px is None or py is None:
            px, py = depot[0], depot[1]

        rx = safe_float(r.get("x", None), None)
        ry = safe_float(r.get("y", None), None)
        if rx is None or ry is None:
            return False

        travel_to = dist((px, py), (rx, ry)) / max(safe_float(V["speed"], 0.0), 1e-9)
        arrive = t_ref + travel_to
        start = max(arrive, safe_float(r.get("e_i", 0.0), 0.0))
        if start > safe_float(r.get("l_i", 0.0), 0.0) + 1e-12:
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

    def vehicle_eta_lower_bound(k, rid, t_now):
        """Lower-bound ETA for ordering vehicles (ignore feasibility nuances)."""
        V = vehs[int(k)]
        r = req_df.loc[rid]
        rx = safe_float(r.get("x", None), None)
        ry = safe_float(r.get("y", None), None)
        if rx is None or ry is None:
            return float("inf")

        pos_v = V.get("pos", depot)
        if isinstance(pos_v, (tuple, list)) and len(pos_v) >= 2:
            px = safe_float(pos_v[0], None)
            py = safe_float(pos_v[1], None)
        else:
            px = py = None
        if px is None or py is None:
            px, py = depot

        speed_v = max(safe_float(V.get("speed", max_speed_all)), 1e-9)
        travel = dist((px, py), (rx, ry)) / speed_v
        return max(safe_float(t_now, 0.0), safe_float(V.get("free_at", t_now), safe_float(t_now, 0.0))) + travel

    def earliest_possible_service_lower_bound(rid, t_now):
        """
        ETA lower bound = max(t_now, free_at) + dist(current_pos, req)/speed
        uncertain=True chỉ khi KHÔNG tính được eta cho bất kỳ xe nào (thật sự thiếu dữ liệu hợp lệ).
        """
        r = req_df.loc[rid]
        rx = safe_float(r.get("x", None), None)
        ry = safe_float(r.get("y", None), None)
        if rx is None or ry is None:
            diag["eta_invalid_req_xy"] = diag.get("eta_invalid_req_xy", 0) + 1
            return None, True

        best = None
        bad = 0

        for V in vehs:
            free = max(safe_float(t_now, 0.0), safe_float(V.get("free_at", t_now), safe_float(t_now, 0.0)))

            speed_v = safe_float(V.get("speed", None), None)
            if speed_v is None or speed_v <= 0:
                speed_v = max_speed_all
            speed_v = max(speed_v, 1e-9)

            pos_v = V.get("pos", depot)
            if isinstance(pos_v, (tuple, list)) and len(pos_v) >= 2:
                px = safe_float(pos_v[0], None)
                py = safe_float(pos_v[1], None)
            else:
                px = py = None
            if px is None or py is None:
                px, py = depot[0], depot[1]

            try:
                travel = dist((px, py), (rx, ry)) / speed_v
                eta = free + travel
            except Exception:
                bad += 1
                continue

            best = eta if best is None else min(best, eta)

        if best is None:
            diag["eta_all_failed"] = diag.get("eta_all_failed", 0) + 1
            return None, True

        if bad > 0:
            diag["eta_partial_failed"] = diag.get("eta_partial_failed", 0) + 1

        return best, False

    def evaluate_global_infeasible(rid, t_now):
        """
        Return {"drop": bool, "uncertain": bool, "min_due_left": float, "travel_lb": float}
        Drop only when lower-bound ETA + service_time already misses l_i for every vehicle
        and the system is idle (no backlog uncertainty).
        """
        r = req_df.loc[rid]
        rx = safe_float(r.get("x", None), None)
        ry = safe_float(r.get("y", None), None)
        if rx is None or ry is None:
            return {"drop": False, "uncertain": True, "min_due_left": float("inf"), "travel_lb": float("inf")}

        service_time = safe_float(r.get("service_time", 0.0), 0.0)
        eta_list = []
        travel_bounds = []
        uncertain = False

        for V in vehs:
            pos_v = V.get("pos", depot)
            if isinstance(pos_v, (tuple, list)) and len(pos_v) >= 2:
                px = safe_float(pos_v[0], None)
                py = safe_float(pos_v[1], None)
            else:
                px = py = None
            if px is None or py is None:
                px, py = depot

            speed_v = max(safe_float(V.get("speed", max_speed_all)), 1e-9)

            try:
                travel_lb = dist((px, py), (rx, ry)) / speed_v
            except Exception:
                uncertain = True
                continue

            eta_lb = max(safe_float(t_now, 0.0), safe_float(V.get("free_at", t_now), safe_float(t_now, 0.0))) + travel_lb + service_time
            eta_list.append(eta_lb)
            travel_bounds.append(travel_lb)

        due = safe_float(r.get("l_i", 0.0), 0.0)
        min_due_left = due - safe_float(t_now, 0.0)
        travel_lb_min = min(travel_bounds) if travel_bounds else float("inf")

        idle_queues = not any(len(v.get("queue", [])) > 0 for v in vehs)
        idle_time = not any(safe_float(v.get("free_at", 0.0), 0.0) > safe_float(t_now, 0.0) + EPS for v in vehs)
        pending_small = len(pending) <= 3

        if min_due_left > 2.0 * travel_lb_min:
            uncertain = True
        if not (idle_queues and idle_time and pending_small):
            uncertain = True
        if not eta_list:
            uncertain = True

        all_late = eta_list and all(eta > due - EPS for eta in eta_list)

        return {
            "drop": bool(all_late and (not uncertain)),
            "uncertain": bool(uncertain),
            "min_due_left": float(min_due_left),
            "travel_lb": float(travel_lb_min),
        }

    def drop_globally_infeasible(t_now):
        """
        Drop an toàn: chỉ drop khi lower bound ETA chắc chắn > l_i - EPS
        và không có queue/backlog bận (để tránh drop nhầm).
        """
        if not pending:
            return

        for rid in list(pending):
            diag["glob_infeasible_checks"] = diag.get("glob_infeasible_checks", 0) + 1
            decision = evaluate_global_infeasible(rid, t_now)
            if decision.get("uncertain"):
                diag["glob_infeasible_uncertain"] = diag.get("glob_infeasible_uncertain", 0) + 1
                continue
            if decision.get("drop"):
                pending.discard(rid)
                dropped.add(rid)
                diag["glob_infeasible_drops"] = diag.get("glob_infeasible_drops", 0) + 1
                log_drop(rid, "globally_infeasible_due", t_now=t_now)

    def assign_request(req_idx, t_now, veh_hint=None, pos_hint=None):
        VEH_TOP_K = 6  # thử tối đa K xe theo ETA lower bound
        candidates = []

        if veh_hint is not None:
            k = int(veh_hint)
            if 0 <= k < len(vehs):
                ins = None
                if pos_hint is not None:
                    pos = int(pos_hint)
                    q = list(vehs[k]["queue"])
                    pos = max(0, min(pos, len(q)))
                    seq = q[:pos] + [req_idx] + q[pos:]
                    t_plan = max(safe_float(t_now, 0.0), safe_float(vehs[k]["free_at"], 0.0))
                    if not spend_sim_budget():
                        res = {"feasible": False, "reason": "sim_budget_exhausted"}
                    else:
                        res = _simulate_route(seq, vehs[k], t_plan, depot, req_df, depot, Lw, load_start=0.0)
                    if res.get("feasible"):
                        ins = {
                            "pos": pos, "score": float(res["end_time"]), "end_time": float(res["end_time"]),
                            "slack_min": float(res.get("slack_min", 0.0)), "slack_risk": float(res.get("slack_risk", 0.0)),
                            "due": float(req_df.loc[req_idx, "l_i"]),
                        }

                if ins is None:
                    ins = best_insertion_with_budget(req_idx, vehs[k], vehs[k]["queue"], t_now, req_df, depot, Lw, base_cache, diag)

                if ins:
                    candidates.append((k, ins))

        if not candidates:
            eta_sorted = []
            for k, V in enumerate(vehs):
                eta_lb = vehicle_eta_lower_bound(k, req_idx, t_now)
                eta_sorted.append((eta_lb, k))
            eta_sorted = sorted(eta_sorted, key=lambda x: x[0])[:VEH_TOP_K]
            for _, k in eta_sorted:
                if tick_budget["remaining"] <= 0:
                    break
                V = vehs[k]
                ins = best_insertion_with_budget(req_idx, V, V["queue"], t_now, req_df, depot, Lw, base_cache, diag)
                if ins:
                    candidates.append((k, ins))

        n_feasible = len(candidates)
        if diag is not None:
            diag["feasible_vehicle_count_sum"] = diag.get("feasible_vehicle_count_sum", 0) + n_feasible
            diag["feasible_vehicle_count_calls"] = diag.get("feasible_vehicle_count_calls", 0) + 1

        if not candidates:
            pending.add(req_idx)
            diag["count_no_feasible_insertion_on_arrival"] = diag.get("count_no_feasible_insertion_on_arrival", 0) + 1
            return False

        # chọn theo rank_key, tie-break bằng regret_vehicle (score khoảng cách top2)
        candidates_sorted = sorted(candidates, key=lambda x: x[1].get("rank_key", (float("inf"),)))
        best_k, best_ins = candidates_sorted[0]
        regret_vehicle = None
        if len(candidates_sorted) > 1:
            second_ins = candidates_sorted[1][1]
            regret_vehicle = safe_float(second_ins.get("score", 0.0) - best_ins.get("score", 0.0), None)
        if diag is not None and regret_vehicle is not None:
            diag["regret_vehicle_sum"] = diag.get("regret_vehicle_sum", 0.0) + regret_vehicle
            diag["regret_vehicle_calls"] = diag.get("regret_vehicle_calls", 0) + 1

        V = vehs[best_k]
        V["queue"].insert(int(best_ins["pos"]), req_idx)
        need_repair = (safe_float(best_ins.get("slack_min", 1e9), 1e9) < 0.2 * max(Lw, 1.0)) or \
                      (safe_float(best_ins.get("delta_end", 0.0), 0.0) / max(Lw, 1.0) > 0.02)
        max_moves = 8 if need_repair else 0
        if diag is not None and need_repair:
            diag["repair_called"] = diag.get("repair_called", 0) + 1
        new_q = _local_repair(V["queue"], V, t_now, req_df, depot, Lw, center_pos=int(best_ins["pos"]), radius=2, max_moves=max_moves)
        if diag is not None and need_repair:
            if new_q != V["queue"]:
                diag["repair_accepted"] = diag.get("repair_accepted", 0) + 1
        V["queue"] = new_q
        return True

    def pull_from_pending_fast(veh_idx, t_now, max_pull=20, cand_limit=20, max_sim_calls_pull=180):
        if (not pending) or tick_budget["remaining"] <= 0:
            return

        diag["pull_calls"] = diag.get("pull_calls", 0) + 1
        V = vehs[int(veh_idx)]

        def pending_key(i):
            r = req_df.loc[i]
            rx = safe_float(r.get("x", None), None)
            ry = safe_float(r.get("y", None), None)
            if rx is None or ry is None:
                return (float("inf"), float("inf"), float("inf"))
            travel_dep = dist(depot, (rx, ry)) / max_speed_all
            margin = safe_float(r["l_i"], 0.0) - max(safe_float(t_now, 0.0) + travel_dep, safe_float(r["e_i"], 0.0))
            age = safe_float(t_now, 0.0) - safe_float(r["t_arrive"], 0.0)
            return (safe_float(r["l_i"], 0.0), -margin, -age)

        cand_ids = sorted(list(pending), key=pending_key)[:int(cand_limit)]
        limit = int(max_pull)

        sim_calls = 0
        pulled = 0

        while pulled < limit and sim_calls < int(max_sim_calls_pull) and tick_budget["remaining"] > 0:
            best_pick = None
            for rid in cand_ids:
                if tick_budget["remaining"] <= 0 or sim_calls >= int(max_sim_calls_pull):
                    break
                if (safe_float(t_now, 0.0) - safe_float(last_try_time.get(rid, -1e9), -1e9)) < 30.0:
                    continue
                ins = best_insertion_with_budget(rid, V, V["queue"], t_now, req_df, depot, Lw, base_cache, diag)
                sim_calls += 1
                last_try_time[rid] = float(t_now)

                if not ins:
                    continue

                cand = (ins["score"], ins.get("due", float("inf")), -ins.get("slack_min", 0.0), rid, ins)
                if (best_pick is None) or (cand < (best_pick[0]["score"], best_pick[0].get("due", float("inf")), -best_pick[0].get("slack_min", 0.0), best_pick[1], best_pick[0])):
                    best_pick = (ins, rid)

            if not best_pick or sim_calls >= int(max_sim_calls_pull) or tick_budget["remaining"] <= 0:
                break

            ins, rid = best_pick
            V["queue"].insert(int(ins["pos"]), rid)
            V["queue"] = _local_repair(V["queue"], V, t_now, req_df, depot, Lw, center_pos=int(ins["pos"]), radius=2)
            pending.discard(rid)
            cand_ids = [i for i in cand_ids if i != rid]
            pulled += 1

        diag["pull_sim_calls_total"] = diag.get("pull_sim_calls_total", 0) + sim_calls
        diag["avg_sim_calls_per_pull"] = diag["pull_sim_calls_total"] / max(diag.get("pull_calls", 1), 1)

    def opportunistic_assign_pending(t_now, eps=1e-6):
        if (not pending) or tick_budget["remaining"] <= 0:
            return
        cooldown_soft = 25.0
        last = safe_float(diag.get("last_pending_assign_time", 0.0), 0.0)
        if (safe_float(t_now, 0.0) - last) < (cooldown_soft - eps):
            return
        diag["last_pending_assign_time"] = float(t_now)
        diag["opp_calls"] = diag.get("opp_calls", 0) + 1

        def urgency_key(rid):
            r = req_df.loc[rid]
            due_left = safe_float(r.get("l_i", 0.0), 0.0) - safe_float(t_now, 0.0)
            slack = safe_float(r.get("l_i", 0.0), 0.0) - max(safe_float(t_now, 0.0), safe_float(r.get("e_i", 0.0), 0.0))
            return (due_left, slack)

        pending_sorted = sorted(list(pending), key=urgency_key)
        core_k = 5 if len(pending_sorted) > 3 else min(4, len(pending_sorted))
        req_candidates = pending_sorted[:core_k]

        diversity = []
        for rid in pending_sorted[core_k:]:
            if random.random() < 0.2:
                diversity.append(rid)
            if len(diversity) >= 2:
                break
        req_candidates.extend(diversity)

        sim_budget = 100
        sim_used = 0
        # Stall-guard: stop this tick after a few consecutive failed insertions.
        stall_fail = 0
        veh_top = 4

        for rid in req_candidates:
            if tick_budget["remaining"] <= 0 or sim_used >= sim_budget or stall_fail >= 3:
                break
            last_gap = safe_float(t_now, 0.0) - safe_float(last_try_time.get(rid, -1e9), -1e9)
            due_left = safe_float(req_df.loc[rid, "l_i"], 0.0) - safe_float(t_now, 0.0)
            if (last_gap < cooldown_soft) and (due_left > cooldown_soft):
                continue

            veh_candidates = sorted(list(range(len(vehs))), key=lambda k: vehicle_eta_lower_bound(k, rid, t_now))[:veh_top]
            success = False

            for k in veh_candidates:
                if tick_budget["remaining"] <= 0 or sim_used >= sim_budget:
                    break
                ins = best_insertion_with_budget(rid, vehs[k], vehs[k].get("queue", []), t_now, req_df, depot, Lw, base_cache, diag)
                sim_used += 1
                last_try_time[rid] = float(t_now)
                if not ins:
                    continue
                vehs[k]["queue"].insert(int(ins["pos"]), rid)
                vehs[k]["queue"] = _local_repair(vehs[k]["queue"], vehs[k], t_now, req_df, depot, Lw, center_pos=int(ins["pos"]), radius=2)
                pending.discard(rid)
                diag["opp_success"] = diag.get("opp_success", 0) + 1
                success = True
                stall_fail = 0
                break

            if not success:
                stall_fail += 1
                if stall_fail >= 3:
                    diag["stall_breaks"] = diag.get("stall_breaks", 0) + 1
                    diag["opp_stall_breaks"] = diag.get("opp_stall_breaks", 0) + 1
                    break

    def opportunistic_assign_pending_backlog(t_now):
        if len(pending) < 10 or tick_budget["remaining"] <= 0:
            return
        diag["opp_backlog_calls"] = diag.get("opp_backlog_calls", 0) + 1

        def urgency_key(i):
            r = req_df.loc[i]
            due_left = safe_float(r.get("l_i", 0.0), 0.0) - safe_float(t_now, 0.0)
            slack = safe_float(r.get("l_i", 0.0), 0.0) - max(safe_float(t_now, 0.0), safe_float(r.get("e_i", 0.0), 0.0))
            return (due_left, slack)

        pending_sorted = sorted(list(pending), key=urgency_key)
        req_candidates = pending_sorted[:5]
        due_floor = float(req_df.loc[pending_sorted[0], "l_i"]) if pending_sorted else None

        sim_budget = 150
        sim_used = 0
        veh_top = min(8, len(vehs))

        for rid in req_candidates:
            if sim_used >= sim_budget or tick_budget["remaining"] <= 0:
                break
            if (safe_float(t_now, 0.0) - safe_float(last_try_time.get(rid, -1e9), -1e9)) < 30.0:
                continue

            scored = []
            for k, V in enumerate(vehs):
                if sim_used >= sim_budget or tick_budget["remaining"] <= 0:
                    break
                ins = best_insertion_with_budget(
                    rid, V, V.get("queue", []), t_now, req_df, depot, Lw, base_cache, diag,
                    positions_override=12, due_floor=due_floor, backlog_active=True
                )
                sim_used += 1
                last_try_time[rid] = float(t_now)
                if not ins:
                    continue
                scored.append((ins["score"], ins.get("due", float("inf")), -ins.get("slack_min", 0.0), k, ins))

            if not scored:
                continue
            scored.sort()

            for _, _, _, k, ins in scored[:veh_top]:
                V = vehs[k]
                V["queue"].insert(int(ins["pos"]), rid)
                V["queue"] = _local_repair(V["queue"], V, t_now, req_df, depot, Lw, center_pos=int(ins["pos"]), radius=2)
                pending.discard(rid)
                diag["opp_backlog_success"] = diag.get("opp_backlog_success", 0) + 1
                break

    def emergency_assign_pending(t_now):
        if (not pending) or tick_budget["remaining"] <= 0:
            return
        min_due_left = min([safe_float(req_df.loc[i, "l_i"], 0.0) - safe_float(t_now, 0.0) for i in pending])
        if min_due_left > max(90.0, 0.1 * max(float(Lw), 1.0)):
            return

        rid = min(list(pending), key=lambda i: safe_float(req_df.loc[i, "l_i"], 0.0))
        veh_candidates = sorted(range(len(vehs)), key=lambda k: vehicle_eta_lower_bound(k, rid, t_now))[:5]

        sim_budget = 220
        sim_used = 0

        diag["emergency_calls"] = diag.get("emergency_calls", 0) + 1
        diag["emergency_eject_calls"] = diag.get("emergency_eject_calls", 0) + 1

        for k in veh_candidates:
            if sim_used >= sim_budget or tick_budget["remaining"] <= 0:
                return
            queue_now = list(vehs[k].get("queue", []))

            ins = best_insertion_with_budget(rid, vehs[k], queue_now, t_now, req_df, depot, Lw, base_cache, diag, positions_override=25)
            sim_used += 1
            if ins:
                vehs[k]["queue"] = queue_now[:ins["pos"]] + [rid] + queue_now[ins["pos"]:]
                pending.discard(rid)
                diag["emergency_success"] = diag.get("emergency_success", 0) + 1
                return
            if sim_used >= sim_budget or tick_budget["remaining"] <= 0:
                return

            victims = sorted(queue_now, key=lambda i: safe_float(req_df.loc[i, "l_i"], 0.0), reverse=True)[:3]
            for victim in victims:
                if sim_used >= sim_budget or tick_budget["remaining"] <= 0:
                    return
                seq2 = [x for x in queue_now if x != victim]
                ins2 = best_insertion_with_budget(rid, vehs[k], seq2, t_now, req_df, depot, Lw, base_cache, diag, positions_override=25)
                sim_used += 1
                if ins2:
                    vehs[k]["queue"] = seq2[:ins2["pos"]] + [rid] + seq2[ins2["pos"]:]
                    pending.discard(rid)
                    pending.add(victim)
                    diag["emergency_success"] = diag.get("emergency_success", 0) + 1
                    diag["emergency_eject_success"] = diag.get("emergency_eject_success", 0) + 1
                    return
                if sim_used >= sim_budget or tick_budget["remaining"] <= 0:
                    return

            if len(queue_now) >= 2:
                victims = sorted(queue_now, key=lambda i: safe_float(req_df.loc[i, "l_i"], 0.0), reverse=True)[:4]
                for i in range(len(victims)):
                    for j in range(i + 1, len(victims)):
                        if sim_used >= sim_budget or tick_budget["remaining"] <= 0:
                            return
                        pair = (victims[i], victims[j])
                        seq3 = [x for x in queue_now if x not in pair]
                        ins3 = best_insertion_with_budget(rid, vehs[k], seq3, t_now, req_df, depot, Lw, base_cache, diag, positions_override=25)
                        sim_used += 1
                        if ins3:
                            vehs[k]["queue"] = seq3[:ins3["pos"]] + [rid] + seq3[ins3["pos"]:]
                            pending.discard(rid)
                            pending.add(pair[0])
                            pending.add(pair[1])
                            diag["emergency_success"] = diag.get("emergency_success", 0) + 1
                            diag["emergency_eject_success"] = diag.get("emergency_eject_success", 0) + 1
                            return
                        if sim_used >= sim_budget:
                            return

    def urgent_pull_pending(t_now, max_candidates=5, veh_top_k=3, sim_budget=120):
        if (not pending) or tick_budget["remaining"] <= 0:
            return
        min_due_left = min([safe_float(req_df.loc[i, "l_i"], 0.0) - safe_float(t_now, 0.0) for i in pending])
        threshold = max(60.0, 0.1 * max(float(Lw), 1.0))
        if min_due_left > threshold:
            return

        diag["urgent_pull_calls"] = diag.get("urgent_pull_calls", 0) + 1
        sim_used = 0

        urgent_set = sorted(list(pending), key=lambda i: safe_float(req_df.loc[i, "l_i"], 0.0))[:int(max_candidates)]
        for rid in urgent_set:
            if tick_budget["remaining"] <= 0 or sim_used >= int(sim_budget):
                break
            if (safe_float(t_now, 0.0) - safe_float(last_try_time.get(rid, -1e9), -1e9)) < 30.0:
                continue

            veh_scores = []
            veh_order = sorted(range(len(vehs)), key=lambda k: vehicle_eta_lower_bound(k, rid, t_now))[:max(int(veh_top_k) + 1, 4)]
            for k in veh_order:
                V = vehs[k]
                if safe_float(V["free_at"], 0.0) > safe_float(t_now, 0.0):
                    continue
                if sim_used >= int(sim_budget):
                    break
                if tick_budget["remaining"] <= 0:
                    break
                ins = best_insertion_with_budget(rid, V, V.get("queue", []), t_now, req_df, depot, Lw, base_cache, diag)
                sim_used += 1
                last_try_time[rid] = float(t_now)
                if not ins:
                    continue
                veh_scores.append((ins["score"], -ins.get("slack_min", 0.0), k, ins))

            if sim_used >= int(sim_budget) or tick_budget["remaining"] <= 0:
                break
            if not veh_scores:
                continue

            veh_scores.sort()
            for _, _, k, ins in veh_scores[:int(veh_top_k)]:
                V = vehs[k]
                V["queue"].insert(int(ins["pos"]), rid)
                V["queue"] = _local_repair(V["queue"], V, t_now, req_df, depot, Lw, center_pos=int(ins["pos"]), radius=2)
                pending.discard(rid)
                diag["urgent_pull_success"] = diag.get("urgent_pull_success", 0) + 1
                break

        diag["sim_calls_urgent_total"] = diag.get("sim_calls_urgent_total", 0) + sim_used

    def expire_list(lst, t_now, reason, veh_idx=None):
        keep = []
        for i in lst:
            if safe_float(t_now, 0.0) >= safe_float(req_df.loc[i, "l_i"], 0.0) - EPS:
                dropped.add(i)
                log_drop(i, reason, t_now=t_now, veh_idx=veh_idx)
            else:
                keep.append(i)
        return keep

    # ----------------------
    # Main loop
    # ----------------------
    prev_signature = None
    decision_dt = 30.0

    while True:
        reset_tick_budget()
        all_done = (len(served) + len(dropped) >= N) and all(len(v["queue"]) == 0 for v in vehs) and (len(pending) == 0)
        if all_done:
            break

        next_arrival = safe_float(req_df.loc[next_idx, "t_arrive"], np.inf) if next_idx < N else np.inf
        free_candidates = [safe_float(v["free_at"], np.inf) for v in vehs if safe_float(v["free_at"], 0.0) > safe_float(t, 0.0) + EPS]
        next_free = min(free_candidates) if free_candidates else np.inf
        pending_future = [safe_float(req_df.loc[i, "l_i"], np.inf) for i in pending if safe_float(req_df.loc[i, "l_i"], 0.0) >= safe_float(t, 0.0) - EPS]
        next_due_pending = min(pending_future) if pending_future else np.inf

        need_decision_tick = len(pending) > 0
        if need_decision_tick and next_decision_time is None:
            next_decision_time = safe_float(t, 0.0) + decision_dt
        next_decision = safe_float(next_decision_time, np.inf) if next_decision_time is not None else np.inf

        t_next = min(next_arrival, next_free, next_due_pending, next_decision)
        if t_next == np.inf:
            break

        if t_next > safe_float(t, 0.0) + EPS:
            t = float(t_next)

        diag["pending_size_over_time"].append(len(pending))
        if pending:
            diag["min_due_remaining"].append(min([safe_float(req_df.loc[i, "l_i"], 0.0) - safe_float(t, 0.0) for i in pending]))
            drop_globally_infeasible(t)

        arrived_mask = (req_df["t_arrive"] <= safe_float(t, 0.0)) & (~req_df.index.isin(served)) & (~req_df.index.isin(dropped))
        pending_coords = list(req_df.loc[arrived_mask, ["x", "y"]].itertuples(index=False, name=None))

        state_common = {
            "time": float(t),
            "vehicles": vehs,
            "req_df": req_df,
            "depot": depot,
            "D_max": D_max,
            "H": H,
            "Q_max": Q_max,
            "pending_coords": pending_coords,
            "is_feasible": is_feasible,
            "total_demand": float(total_demand_all),
            "total_reqs": int(N),
            "feature_mask": cfg.get("features", {}).get("mask"),
        }

        # (1) arrivals
        while next_idx < N and safe_float(req_df.loc[next_idx, "t_arrive"], np.inf) <= safe_float(t, 0.0) + EPS:
            state_common["time"] = float(t)

            ins_map = {}
            for k, V in enumerate(vehs):
                if tick_budget["remaining"] <= 0:
                    break
                info = best_insertion_with_budget(next_idx, V, V.get("queue", []), t, req_df, depot, Lw, base_cache, diag)
                if info:
                    ins_map[k] = info
            state_common["insertion_info"] = ins_map

            r_new = req_df.loc[next_idx]
            diag["glob_infeasible_checks"] = diag.get("glob_infeasible_checks", 0) + 1
            decision = evaluate_global_infeasible(next_idx, t)
            if decision.get("uncertain"):
                diag["glob_infeasible_uncertain"] = diag.get("glob_infeasible_uncertain", 0) + 1
            elif decision.get("drop"):
                diag["glob_infeasible_drops"] = diag.get("glob_infeasible_drops", 0) + 1
                dropped.add(next_idx)
                log_drop(next_idx, "globally_infeasible_due", t_now=t)
                next_idx += 1
                continue

            choice = policy.route_request(state_common, next_idx)
            veh_hint, pos_hint = normalize_choice(choice)

            assigned = False
            if veh_hint is not None:
                assigned = assign_request(next_idx, t, veh_hint=veh_hint, pos_hint=pos_hint)
            if not assigned:
                assign_request(next_idx, t)

            next_idx += 1

        # (1b) expire pending/queue
        for rid in list(pending):
            if safe_float(t, 0.0) >= safe_float(req_df.loc[rid, "l_i"], 0.0) - EPS:
                pending.discard(rid)
                dropped.add(rid)
                log_drop(rid, "pending_expired", t_now=t)

        for k, V in enumerate(vehs):
            V["queue"] = expire_list(V.get("queue", []), t, "expired_waiting_in_queue", veh_idx=k)

        # (1c) emergency/backlog/urgent
        emergency_assign_pending(t_now=t)
        if pending:
            opportunistic_assign_pending_backlog(t_now=t)
            urgent_pull_pending(t_now=t)

        # (1d) decision tick
        if need_decision_tick and safe_float(t, 0.0) >= safe_float(next_decision, np.inf) - EPS:
            diag["decision_ticks"] = diag.get("decision_ticks", 0) + 1
            next_decision_time = safe_float(t, 0.0) + decision_dt
            drop_globally_infeasible(t)
            emergency_assign_pending(t_now=t)
            opportunistic_assign_pending(t_now=t)
            opportunistic_assign_pending_backlog(t_now=t)
            urgent_pull_pending(t_now=t)
        else:
            if pending:
                opportunistic_assign_pending(t_now=t)

        if not need_decision_tick:
            next_decision_time = None

        # (2) free vehicles pull
        for k, V in enumerate(vehs):
            if safe_float(V["free_at"], 0.0) > safe_float(t, 0.0):
                continue
            pull_from_pending_fast(k, t, max_pull=20, cand_limit=20, max_sim_calls_pull=180)

        # (3) serve queues
        for k, V in enumerate(vehs):
            if safe_float(V["free_at"], 0.0) > safe_float(t, 0.0) or not V.get("queue"):
                continue

            t_v = max(safe_float(t, 0.0), safe_float(V["free_at"], 0.0))
            tour_start = float(t_v)
            pos_v = V["pos"]
            load_v = safe_float(V["load"], 0.0)

            while V.get("queue"):
                V["queue"] = expire_list(V["queue"], t_v, "expired_waiting_in_queue", veh_idx=k)
                if not V["queue"]:
                    break

                state_common["time"] = float(t_v)
                nxt_hint = policy.select_next(state_common, k)
                nxt = nxt_hint if nxt_hint in V["queue"] else V["queue"][0]
                r = req_df.loc[nxt]

                if safe_float(r["t_arrive"], np.inf) > safe_float(t_v, 0.0):
                    break

                if not is_feasible(k, r, load_override=load_v, t_now=t_v, pos_override=pos_v):
                    V["queue"] = [x for x in V["queue"] if x != nxt]
                    continue

                rx = safe_float(r.get("x", None), None)
                ry = safe_float(r.get("y", None), None)
                if rx is None or ry is None:
                    V["queue"] = [x for x in V["queue"] if x != nxt]
                    continue

                speed = max(safe_float(V["speed"], 0.0), 1e-9)
                travel_to = dist(pos_v, (rx, ry)) / speed
                arrive_cust = safe_float(t_v, 0.0) + travel_to
                start = max(arrive_cust, safe_float(r["e_i"], 0.0))

                if start > safe_float(r["l_i"], 0.0) + 1e-12:
                    dropped.add(nxt)
                    log_drop(nxt, "cannot_reach_before_due", t_now=t_v, veh_idx=k)
                    V["queue"] = [x for x in V["queue"] if x != nxt]
                    continue

                remaining = [req for req in V["queue"] if req != nxt]
                veh_now = dict(V)
                veh_now["load"] = float(load_v)
                veh_now["pos"] = pos_v
                sim_seq = [nxt] + remaining
                res_future = _simulate_route(sim_seq, veh_now, t_v, pos_v, req_df, depot, Lw)
                if not res_future.get("feasible"):
                    pending.add(nxt)
                    V["queue"] = [x for x in V["queue"] if x != nxt]
                    continue

                # Commit serve
                t_v = float(start)
                pos_v = (rx, ry)
                load_v += safe_float(r["demand"], 0.0)

                served.add(nxt)
                V["queue"] = [x for x in V["queue"] if x != nxt]
                timeline.append({"vehicle": int(k), "req": int(nxt), "start": float(start), "finish": float(start), "action": "serve"})

            # end tour -> depot
            travel_back = dist(pos_v, depot) / max(safe_float(V["speed"], 0.0), 1e-9)
            V["free_at"] = float(t_v) + travel_back
            V["pos"] = depot
            V["load"] = 0.0

            # Drop requests that expire before returning depot
            keep = []
            for i in V.get("queue", []):
                if safe_float(V["free_at"], 0.0) >= safe_float(req_df.loc[i, "l_i"], 0.0) - EPS:
                    dropped.add(i)
                    log_drop(i, "expired_in_queue_after_tour", t_now=float(V["free_at"]), veh_idx=k)
                else:
                    keep.append(i)
            V["queue"] = keep

        # (4) expire pending after service/pull
        for rid in list(pending):
            if safe_float(t, 0.0) >= safe_float(req_df.loc[rid, "l_i"], 0.0) - EPS:
                pending.discard(rid)
                dropped.add(rid)
                log_drop(rid, "pending_expired", t_now=t)

        if pending:
            diag.setdefault("pending_snapshot", []).append({
                "time": float(t),
                "pending": int(len(pending)),
                "min_due_left": float(min([safe_float(req_df.loc[i, "l_i"], 0.0) - safe_float(t, 0.0) for i in pending])),
            })

        for k, V in enumerate(vehs):
            V["queue"] = expire_list(V.get("queue", []), t, "expired_waiting_in_queue", veh_idx=k)

        # Stall guard
        signature = (
            round(float(t), 6),
            int(next_idx),
            int(len(pending)),
            int(len(served)),
            int(len(dropped)),
            int(sum(len(v.get("queue", [])) for v in vehs)),
            tuple(round(float(v["free_at"]), 6) for v in vehs),
        )
        if signature == prev_signature:
            diag["stall_breaks"] = diag.get("stall_breaks", 0) + 1
            future_candidates = [x for x in [next_arrival, next_free, next_due_pending, next_decision] if x > float(t) + EPS]
            if future_candidates:
                t = float(min(future_candidates))
            else:
                t = float(t) + decision_dt
        prev_signature = signature

    # finalize unserved
    for i in range(N):
        if i not in served and i not in dropped:
            dropped.add(i)
            log_drop(i, "unserved_end_of_sim", t_now=t)

    makespan = max([safe_float(v["free_at"], 0.0) for v in vehs] + [0.0])
    stats = {
        "makespan": float(makespan),
        "served": int(len(served)),
        "total": int(N),
        "dropped": int(len(dropped)),
        "drop_reasons": drop_reasons,
        "drop_breakdown": {r: sum(1 for d in drop_reasons if d["reason"] == r) for r in set(d["reason"] for d in drop_reasons)},
        "diag": diag,
    }
    return stats, timeline
