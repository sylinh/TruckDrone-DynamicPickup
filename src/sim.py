import numpy as np
import pandas as pd
from math import hypot
from copy import deepcopy


def dist(a, b): return float(hypot(a[0]-b[0], a[1]-b[1]))


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
    dr = cfg["vehicles"].get("drones", {"count":0})
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
    """Event-driven mô phỏng theo mã giả: request mới -> route R; xe rảnh -> sequence S."""
    req_df = req_df.reset_index(drop=True)
    N = len(req_df)
    depot = tuple(cfg.get("depot", [0.0,0.0]))
    vehs = _mk_vehicles(cfg, depot)
    Q_max = max([v["capacity"] for v in vehs] + [1.0])
    D_max, H = _norm_constants(req_df, cfg, depot)
    Lw = float(cfg.get("constraints", {}).get("Lw", 1e9))

    served, dropped = set(), set()
    timeline = []
    next_idx = 0
    t = 0.0

    def is_feasible(k, r, load_override=None, t_now=None, pos_override=None):
        V = vehs[k]
        demand = float(r["demand"])
        if demand > V["capacity"]:
            return False
        if V["type"] == "drone" and V.get("radius", None) is not None:
            if dist(depot, (r["x"], r["y"])) > V["radius"]:
                return False
        load_now = V["load"] if load_override is None else load_override
        if (load_now + demand) > V["capacity"]:
            return False
        # check time window feasibility from current pos/time
        t_ref = t if t_now is None else t_now
        pos_ref = V["pos"] if pos_override is None else pos_override
        travel_to = dist(pos_ref, (r["x"], r["y"])) / max(V["speed"], 1e-9)
        arrive = t_ref + travel_to
        start = max(arrive, r["e_i"])
        if start > r["l_i"]:
            return False
        return True

    while True:
        all_done = (len(served) + len(dropped) >= N) and all(len(v["queue"])==0 for v in vehs)
        if all_done:
            break

        next_arrival = req_df.loc[next_idx, "t_arrive"] if next_idx < N else np.inf
        # chỉ tính free_at trong tương lai; nếu đã <= t và không có queue thì bỏ qua để tránh kẹt
        free_candidates = []
        for v in vehs:
            if v["free_at"] > t or (v.get("queue") and len(v["queue"]) > 0):
                free_candidates.append(v["free_at"])
        next_free = min(free_candidates) if free_candidates else np.inf
        t_next = min(next_arrival, next_free)
        if t_next == np.inf:
            break
        t = max(t, t_next)

        # pending coords for features (arrived, not served/dropped)
        arrived_mask = (req_df["t_arrive"] <= t) & (~req_df.index.isin(served)) & (~req_df.index.isin(dropped))
        pending_coords = list(req_df.loc[arrived_mask, ["x","y"]].itertuples(index=False, name=None))

        state_common = {
            "time": t, "vehicles": vehs, "req_df": req_df, "depot": depot,
            "D_max": D_max, "H": H, "Q_max": Q_max,
            "pending_coords": pending_coords, "is_feasible": is_feasible,
        }

        # 1) xử lý request mới (t_arrive <= t)
        while next_idx < N and req_df.loc[next_idx, "t_arrive"] <= t:
            r = req_df.loc[next_idx]
            state_common["time"] = t
            choice = policy.route_request(state_common, next_idx)
            if choice is None:
                dropped.add(next_idx)
            else:
                vehs[choice]["queue"].append(next_idx)
            next_idx += 1

        # 2) xe rảnh chọn khách tiếp theo, có thể phục vụ nhiều khách trên một hành trình trước khi về depot
        for k, V in enumerate(vehs):
            if V["free_at"] > t:
                continue  # đang di chuyển
            if not V.get("queue"):
                continue
            t_v = max(t, V["free_at"])
            pos_v = V["pos"]
            load_v = V["load"]
            tour = []  # (req_idx, pick_time)
            while V.get("queue"):
                state_common["time"] = t_v
                nxt = policy.select_next(state_common, k)
                if nxt is None:
                    # drop hết hạn, giữ lại khả thi
                    keep = []
                    for i in V.get("queue", []):
                        r = req_df.loc[i]
                        if t_v > r["l_i"]:
                            dropped.add(i)
                        else:
                            keep.append(i)
                    V["queue"] = keep
                    break
                r = req_df.loc[nxt]
                if nxt in served or nxt in dropped:
                    V["queue"] = [x for x in V["queue"] if x != nxt]
                    continue
                if r["t_arrive"] > t_v:
                    # chưa đến thời điểm request xuất hiện
                    break
                if not is_feasible(k, r, load_override=load_v, t_now=t_v, pos_override=pos_v):
                    V["queue"] = [x for x in V["queue"] if x != nxt]
                    continue

                # thời gian di chuyển từ vị trí hiện tại tới khách
                travel_to = dist(pos_v, (r["x"], r["y"])) / max(V["speed"], 1e-9)
                arrive_cust = t_v + travel_to
                start = max(arrive_cust, r["e_i"])
                if start > r["l_i"]:
                    dropped.add(nxt)
                    V["queue"] = [x for x in V["queue"] if x != nxt]
                    continue

                # kiểm tra Lw cho toàn tour nếu thêm khách này (tính thời gian về depot)
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
                    continue

                # serve khách, ở lại vị trí khách
                t_v = start  # service time ~0
                pos_v = (r["x"], r["y"])
                load_v += float(r["demand"])
                tour.append((nxt, float(t_pick_new)))
                served.add(nxt)
                V["queue"] = [x for x in V["queue"] if x != nxt]
                timeline.append((k, int(nxt), float(start), float(start)))

            # về depot khi không còn queue hoặc không chọn được nữa
            travel_back = dist(pos_v, depot) / max(V["speed"], 1e-9)
            V["free_at"] = t_v + travel_back
            V["pos"] = depot
            V["load"] = 0.0

            # drop các khách còn lại nếu đã quá hạn
            keep = []
            for i in V.get("queue", []):
                r = req_df.loc[i]
                if V["free_at"] > r["l_i"]:
                    dropped.add(i)
                else:
                    keep.append(i)
            V["queue"] = keep

    # drop tất cả request còn lại chưa phục vụ/ chưa drop (không còn sự kiện nào diễn ra)
    for i in range(N):
        if i not in served and i not in dropped:
            dropped.add(i)

    makespan = max([v["free_at"] for v in vehs] + [0.0])
    stats = {"makespan": float(makespan), "served": len(served), "total": N, "dropped": len(dropped)}
    return stats, timeline
