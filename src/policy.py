from abc import ABC, abstractmethod
import numpy as np
from .features import featurize, FEATURE_ORDER


class Policy(ABC):
    """Policy gồm 2 thành phần: routing R (k,i) và sequencing S (k,i)."""

    @abstractmethod
    def route_request(self, state, req_idx):
        """Chọn vehicle cho request mới xuất hiện. Trả về veh_id hoặc None (drop)."""
        ...

    @abstractmethod
    def select_next(self, state, veh_idx):
        """Chọn khách tiếp theo trong queue của vehicle. Trả về req_idx hoặc None (về depot)."""
        ...


class GreedyEDD(Policy):
    """Baseline: gán/ưu tiên theo deadline l_i sớm nhất khả thi."""

    def route_request(self, state, req_idx):
        req = state["req_df"].loc[req_idx]
        best = None
        for k, V in enumerate(state["vehicles"]):
            if not state["is_feasible"](k, req):
                continue
            cand = (req["l_i"], k)
            if best is None or cand < best:
                best = cand
        return None if best is None else best[1]

    def select_next(self, state, veh_idx):
        V = state["vehicles"][veh_idx]
        q = V.get("queue", [])
        if not q:
            return None
        best = None
        for i in q:
            r = state["req_df"].loc[i]
            # check time window feasibility at current time
            now = state["time"]
            if now > r["l_i"]:
                continue
            cand = (r["l_i"], i)
            if best is None or cand < best:
                best = cand
        return None if best is None else best[1]


class GPPolicy(Policy):
    """Hai cây GP: R(features) và S(features) -> tính score, chọn max."""

    def __init__(self, routing_func, sequencing_func):
        self.R = routing_func
        self.S = sequencing_func

    def _score(self, state, veh_idx, req_idx, mode="R"):
        V = state["vehicles"][veh_idx]
        r = state["req_df"].loc[req_idx]
        x = featurize(vehicle=V, req=r, sim=state)
        f = self.R if mode == "R" else self.S
        return float(f(*x))

    def route_request(self, state, req_idx):
        best = None
        for k, V in enumerate(state["vehicles"]):
            if not state["is_feasible"](k, state["req_df"].loc[req_idx]):
                continue
            s = self._score(state, k, req_idx, mode="R")
            if (best is None) or (s > best[0]):
                best = (s, k)
        return None if best is None else best[1]

    def select_next(self, state, veh_idx):
        V = state["vehicles"][veh_idx]
        q = V.get("queue", [])
        if not q:
            return None
        best = None
        for i in q:
            r = state["req_df"].loc[i]
            if state["time"] > r["l_i"]:
                continue
            s = self._score(state, veh_idx, i, mode="S")
            if (best is None) or (s > best[0]):
                best = (s, i)
        return None if best is None else best[1]


def build_baseline():
    return GreedyEDD()
