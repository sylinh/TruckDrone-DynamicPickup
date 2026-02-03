"""
Chính sách chọn xe và thứ tự phục vụ:
- GreedyEDD: baseline ưu tiên deadline sớm nhất trên tập feasible.
- GPPolicy: 2 cây GP cho điểm (veh, req) cho routing/selection.
- MLPolicy: cho điểm bằng model ML đã huấn luyện.
- HybridPolicy: ưu tiên ML, fallback sang policy nền nếu không có lựa chọn.
"""

from abc import ABC, abstractmethod
from .features import featurize, FEATURE_ORDER
from .ml import load_ml_model, score_model
from .ml_logging import DecisionLogger


class Policy(ABC):
    """Policy gồm 2 thành phần: routing R (chọn xe cho request mới) và sequencing S (chọn khách tiếp theo trong queue)."""

    @abstractmethod
    def route_request(self, state, req_idx):
        """Chọn vehicle cho request mới. Trả về veh_id, (veh_id, pos) hoặc None (drop)."""
        ...

    @abstractmethod
    def select_next(self, state, veh_idx):
        """Chọn khách tiếp theo trong queue của vehicle. Trả về req_idx hoặc None (về depot)."""
        ...


class GreedyEDD(Policy):
    """Baseline: ưu tiên deadline l_i sớm nhất (EDD) trên tập feasible."""  # đơn giản, determinisitc

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
            now = state["time"]
            if now > r["l_i"]:
                continue
            cand = (r["l_i"], i)
            if best is None or cand < best:
                best = cand
        return None if best is None else best[1]


class GPPolicy(Policy):
    """Hai cây GP: R(features) và S(features) -> score, chọn max."""  # giữ determinism theo seed khi build cây

    def __init__(self, routing_func, sequencing_func):
        self.R = routing_func
        self.S = sequencing_func
        self.logger = None
        self.instance = None

    def set_logger(self, logger: DecisionLogger | None, instance: str | None = None):
        self.logger = logger
        self.instance = instance

    def _score(self, state, veh_idx, req_idx, mode="R"):
        V = state["vehicles"][veh_idx]
        r = state["req_df"].loc[req_idx]
        sim_state = dict(state)
        sim_state["veh_idx"] = veh_idx
        x = featurize(vehicle=V, req=r, sim=sim_state)
        f = self.R if mode == "R" else self.S
        return float(f(*x))

    def route_request(self, state, req_idx):
        candidates = []
        for k, V in enumerate(state["vehicles"]):
            if not state["is_feasible"](k, state["req_df"].loc[req_idx]):
                continue
            sim_state = dict(state); sim_state["veh_idx"] = k
            feats = featurize(vehicle=V, req=state["req_df"].loc[req_idx], sim=sim_state)
            s = self._score(state, k, req_idx, mode="R")
            candidates.append((s, k, feats))
        if not candidates:
            return None
        best = max(candidates, key=lambda x: x[0])
        if self.logger:
            for s, k, feats in candidates:
                self.logger.log("route", self.instance or "", state["time"], k, req_idx, k == best[1], s, feats)
        return best[1]

    def select_next(self, state, veh_idx):
        V = state["vehicles"][veh_idx]
        q = V.get("queue", [])
        if not q:
            return None
        candidates = []
        for i in q:
            r = state["req_df"].loc[i]
            if state["time"] > r["l_i"]:
                continue
            sim_state = dict(state); sim_state["veh_idx"] = veh_idx
            feats = featurize(vehicle=V, req=r, sim=sim_state)
            s = self._score(state, veh_idx, i, mode="S")
            candidates.append((s, i, feats))
        if not candidates:
            return None
        best = max(candidates, key=lambda x: x[0])
        if self.logger:
            for s, i, feats in candidates:
                self.logger.log("sequence", self.instance or "", state["time"], veh_idx, i, i == best[1], s, feats)
        return best[1]


class MLPolicy(Policy):
    """Score (veh, req) bằng model ML; tách route/sequence nếu model cung cấp."""

    def __init__(self, model_path: str, logger: DecisionLogger | None = None, instance: str | None = None):
        payload = load_ml_model(model_path)
        if isinstance(payload, dict) and "models" in payload:
            self.model_route = payload["models"].get("route")
            self.model_seq = payload["models"].get("sequence", self.model_route)
        else:
            self.model_route = payload
            self.model_seq = payload
        self.logger = logger
        self.instance = instance

    def _score_route(self, feats):
        return score_model({"models": {"route": self.model_route}}, feats)

    def _score_seq(self, feats):
        return score_model({"models": {"route": self.model_seq}}, feats)

    def route_request(self, state, req_idx):
        candidates = []
        for k, V in enumerate(state["vehicles"]):
            if not state["is_feasible"](k, state["req_df"].loc[req_idx]):
                continue
            sim_state = dict(state); sim_state["veh_idx"] = k
            feats = featurize(vehicle=V, req=state["req_df"].loc[req_idx], sim=sim_state)
            s = self._score_route(feats)
            candidates.append((s, k, feats))
        if not candidates:
            return None
        best = max(candidates, key=lambda x: x[0])
        if self.logger:
            for s, k, feats in candidates:
                self.logger.log("route", self.instance or "", state["time"], k, req_idx, k == best[1], s, feats)
        return best[1]

    def select_next(self, state, veh_idx):
        V = state["vehicles"][veh_idx]
        q = V.get("queue", [])
        if not q:
            return None
        candidates = []
        for i in q:
            r = state["req_df"].loc[i]
            if state["time"] > r["l_i"]:
                continue
            sim_state = dict(state); sim_state["veh_idx"] = veh_idx
            feats = featurize(vehicle=V, req=r, sim=sim_state)
            s = self._score_seq(feats)
            candidates.append((s, i, feats))
        if not candidates:
            return None
        best = max(candidates, key=lambda x: x[0])
        if self.logger:
            for s, i, feats in candidates:
                self.logger.log("sequence", self.instance or "", state["time"], veh_idx, i, i == best[1], s, feats)
        return best[1]


class HybridPolicy(Policy):
    """Dùng ML trước; fallback sang policy nền nếu không có lựa chọn khả thi."""

    def __init__(self, ml_policy: MLPolicy, fallback: Policy):
        self.ml_policy = ml_policy
        self.fallback = fallback

    def route_request(self, state, req_idx):
        choice = self.ml_policy.route_request(state, req_idx)
        if choice is None:
            return self.fallback.route_request(state, req_idx)
        return choice

    def select_next(self, state, veh_idx):
        choice = self.ml_policy.select_next(state, veh_idx)
        if choice is None:
            return self.fallback.select_next(state, veh_idx)
        return choice


def build_baseline():
    return GreedyEDD()
