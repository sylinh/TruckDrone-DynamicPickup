from __future__ import annotations
import json, re, sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union
import pandas as pd

REQ_COLS = ["req_id","x","y","demand","drone_ok","t_arrive","e_i","l_i"]

# ---------- JSON/JSONC/JSONL helpers ----------
def _clean_jsonc(text: str) -> str:
    text = text.lstrip("\ufeff")
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return text

def _load_json_relaxed(path: Path) -> Union[dict, list]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    txt = _clean_jsonc(raw)
    try:
        return json.loads(txt)
    except Exception:
        pass
    # try JSONL
    recs: List[dict] = []
    for ln in raw.splitlines():
        s = _clean_jsonc(ln.strip())
        if not s:
            continue
        try:
            recs.append(json.loads(s))
        except Exception:
            continue
    if recs:
        return recs
    # extract biggest json chunk
    m = re.search(r'(\{.*\}|\[.*\])', txt, flags=re.DOTALL)
    if m:
        return json.loads(_clean_jsonc(m.group(1)))
    raise ValueError(f"Cannot parse JSON/JSONC/JSONL: {path}")

# ---------- classify file ----------
def _classify_and_load(path: Path) -> Tuple[str, Union[dict, list]]:
    obj = _load_json_relaxed(path)
    # obvious benchmark: jsonl logs or dict with 'solution'
    if isinstance(obj, list):
        if obj and isinstance(obj[0], dict) and "__" in obj[0]:
            return "benchmark", obj
        if obj and isinstance(obj[0], list) and len(obj[0]) >= 3:
            return "data", obj
        if obj and isinstance(obj[0], dict) and ("x" in obj[0] or "y" in obj[0]):
            return "data", obj
    if isinstance(obj, dict):
        if "solution" in obj or "routes" in obj or "final_routes" in obj:
            return "benchmark", obj
        if "requests" in obj:
            return "data", obj
        # dict-of-arrays?
        arr_keys = {"x","y","demand","demands","dronable","ableServiceByDrone",
                    "r","r_i","request_time","open_time","opentime","e_i","close_time","closetime","l_i"}
        if arr_keys & set(obj.keys()):
            return "data", obj
    return "unknown", obj

# ---------- parse requests (robust) ----------
def _from_dict_of_arrays(d: dict) -> pd.DataFrame:
    def pick(*keys, default=None):
        for k in keys:
            if k in d: return d[k]
        return default
    xs = pick("x") or pick("X")
    ys = pick("y") or pick("Y")
    dem = pick("demand","demands")
    drone = pick("ableServiceByDrone","dronable","drone_ok")
    r = pick("r_i","r","request_time", default=0.0)
    e = pick("open_time","opentime","e_i", default=0.0)
    l = pick("close_time","closetime","l_i", default=1e9)
    if xs is None or ys is None or dem is None:
        raise ValueError("dict-of-arrays missing x/y/demand(s) keys")
    n = len(xs)
    def arr(v, fill=0.0):
        if v is None: return [fill]*n
        return list(v)
    drone = [int(v) for v in arr(drone, 1)]
    rows = []
    for i in range(n):
        rows.append([i, float(xs[i]), float(ys[i]), float(dem[i]),
                     int(drone[i]), float(arr(r)[i]), float(arr(e)[i]), float(arr(l)[i])])
    return pd.DataFrame(rows, columns=REQ_COLS)

def _parse_dynamic_json(json_path: Path, fmt_hint: Optional[str] = None,
                        r_scale_wtw3: Optional[float] = None) -> pd.DataFrame:
    kind, obj = _classify_and_load(json_path)
    if kind != "data":
        raise ValueError(f"{json_path.name} is not a request file (classified: {kind})")

    # normalize to list of rows
    if isinstance(obj, dict):
        if "requests" in obj:
            rows = obj["requests"]
        else:
            # dict-of-arrays variant
            df = _from_dict_of_arrays(obj)
            rows = None
    else:
        rows = obj

    if rows is not None:
        out = []
        for idx, r in enumerate(rows):
            if isinstance(r, dict):
                x = r.get("x"); y = r.get("y")
                demand = r.get("demand", r.get("demands"))
                drone_ok = r.get("ableServiceByDrone", r.get("dronable", r.get("drone_ok", 1)))
                r_i = r.get("r_i", r.get("r", r.get("request_time", 0.0)))
                e_i = r.get("open_time", r.get("opentime", r.get("e_i", 0.0)))
                l_i = r.get("close_time", r.get("closetime", r.get("l_i", e_i)))
                # skip rows missing coordinates
                if x is None or y is None:
                    continue
                out.append([idx, float(x), float(y), float(demand), int(drone_ok),
                            float(r_i), float(e_i), float(l_i)])
            else:
                x,y,demand,drone_ok,r_i,e_i,l_i = r
                out.append([idx, float(x), float(y), float(demand), int(drone_ok),
                            float(r_i), float(e_i), float(l_i)])
        df = pd.DataFrame(out, columns=REQ_COLS)

    # guess format if needed
    if fmt_hint is None:
        parent = json_path.parent.name.lower()
        fmt_hint = "WTW3" if "withtimewindows3" in parent else "WTW"

    # NO scaling by default. Only scale if a numeric factor is explicitly given.
    if fmt_hint and fmt_hint.upper() == "WTW3" and isinstance(r_scale_wtw3, (int, float)) and r_scale_wtw3 not in (None, 1.0):
        df["t_arrive"] = df["t_arrive"] * float(r_scale_wtw3)

    return df

# ---------- benchmark detection + normalization ----------
def _normalize_benchmark(obj: Union[dict, list]) -> dict:
    if isinstance(obj, dict):
        return {"type": "json_object", "raw": obj}

    best_gp = None
    heur = None
    routes: List[dict] = []
    final_routes: dict = {}

    def better(a: dict, b: Optional[dict]) -> bool:
        if b is None: return True
        if a.get("served", -1) != b.get("served", -1):
            return a.get("served", -1) > b.get("served", -1)
        fa, fb = a.get("fitness"), b.get("fitness")
        if fa is not None and fb is not None and abs(fa - fb) > 1e-12:
            return fa < fb
        return a.get("makespan", 1e30) < b.get("makespan", 1e30)

    for r in obj:
        typ = r.get("__"); tag = r.get("_")
        if typ == "HEU" and tag == "heuristic_result":
            res = r.get("result", [None, None])
            heur = {
                "name": r.get("name"),
                "makespan": float(res[0]) if res[0] is not None else None,
                "served": int(res[1]) if res[1] is not None else None,
                "fitness": r.get("fitness"),
            }
        elif typ == "GP" and tag in ("new_gen","full_result"):
            res = r.get("result", [None, None])
            cand = {
                "gen": r.get("gen"),
                "makespan": float(res[0]) if res[0] is not None else None,
                "served": int(res[1]) if res[1] is not None else None,
                "fitness": r.get("fitness"),
                "routing": r.get("routing"),
                "sequencing": r.get("sequencing"),
            }
            if better(cand, best_gp):
                best_gp = cand
        elif tag == "route_log":
            routes.append({
                "type": typ, "vehicle": r.get("vehicle"),
                "route": r.get("route"), "dropped": r.get("dropped", []),
            })
            if typ == "LASTROUTE":
                final_routes[str(r.get("vehicle"))] = {
                    "route": r.get("route"), "dropped": r.get("dropped", []),
                }

    return {"type": "jsonl_log", "summary": {"heuristic": heur, "gp_best": best_gp},
            "routes": routes, "final_routes": final_routes}

def _write_benchmark_cleaned(src_path: Path, inst_dir: Path):
    obj = _load_json_relaxed(src_path)
    norm = _normalize_benchmark(obj)
    (inst_dir / "benchmark.json").write_text(json.dumps(norm, ensure_ascii=False, indent=2), encoding="utf-8")
    if src_path.suffix.lower() == ".jsonc":
        (inst_dir / "benchmark.raw.jsonc").write_text(
            src_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8"
        )

# ---------- static finder ----------
def _find_static_solution(root: Path, stem: str) -> Optional[Path]:
    for cand in ["Static_solution_noTimeWindow","static_solution_noTimeWindow","Static_solution"]:
        d = root / cand
        if not d.exists(): continue
        for ext in (".json",".jsonc",".txt"):
            m = list(d.glob(f"{stem}*{ext}"))
            if m: return sorted(m)[-1]
    return None

# ---------- benchmark matcher ----------
def _find_benchmark(dynamic_json_path: Path) -> Optional[Path]:
    # lấy base id n.m.x ở đầu tên
    name = dynamic_json_path.name
    m = re.match(r"(\d+\.\d+\.\d+)", name)
    base = m.group(1) if m else dynamic_json_path.stem
    pats = [
        f"{base}.json.result.jsonc", f"{base}.result.jsonc", f"{base}*.result.jsonc",
        f"{base}*solution*.jsonc", f"{base}*benchmark*.jsonc", f"{base}*result*.jsonc",
        f"{base}*solution*.json",  f"{base}*benchmark*.json",  f"{base}.sol*.json",
        f"{base}_solution.json",   f"{base}-solution.json",
    ]
    cands: List[Path] = []
    for pat in pats:
        cands += list(dynamic_json_path.parent.glob(pat))
    cands = [p for p in set(cands) if p != dynamic_json_path]
    if not cands: return None

    def score(p: Path):
        nm = p.name.lower(); s = 0
        if "solution" in nm: s += 5
        if "benchmark" in nm: s += 4
        if "result" in nm: s += 3
        if p.suffix.lower() == ".jsonc": s += 1
        return s

    cands.sort(key=score, reverse=True)
    return cands[0]

# ---------- recursive file iterator + natural sort ----------
def _natsort_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', p.name)]

def _iter_source_files(src_path: Path):
    return sorted({*src_path.rglob("*.[Jj][Ss][Oo][Nn]"),
                   *src_path.rglob("*.[Jj][Ss][Oo][Nn][Cc]")},
                  key=_natsort_key)

# ---------- bulk import (dynamic) ----------
def generate_data_from_dynamic_sources(
    src_dirs: Iterable[str] = ("WithTimeWindows","WithTimeWindows3"),
    dst_root: str = "data/raw",
    r_scale_wtw3: Optional[float] = None,
    verbose: bool = False,
) -> List[Tuple[str,int,bool]]:
    dst = Path(dst_root); dst.mkdir(parents=True, exist_ok=True)
    repo_root = Path.cwd()
    report: List[Tuple[str,int,bool]] = []

    for src in src_dirs:
        src_path = Path(src)
        if not src_path.exists():
            sp2 = repo_root / src
            if sp2.exists(): src_path = sp2
        if not src_path.exists():
            print(f"  Skip: '{src}' not found.")
            continue

        if verbose:
            print(f"🔎 Scanning {src_path} ...")

        fmt_hint = "WTW3" if "withtimewindows3" in src_path.name.lower() else "WTW"
        for jp in _iter_source_files(src_path):
            # phân loại sớm để khỏi ép float(None)
            kind, _ = _classify_and_load(jp)

            # id n.m.x để gom output
            m = re.match(r"(\d+\.\d+\.\d+)", jp.name)
            stem = (m.group(1) if m else jp.stem).replace(".json","")
            inst_dir = dst / stem
            inst_dir.mkdir(parents=True, exist_ok=True)

            if kind == "benchmark":
                _write_benchmark_cleaned(jp, inst_dir)
                if verbose: print(f" 🗒  {jp.relative_to(src_path)} → benchmark for {stem}")
                continue
            if kind == "unknown":
                if verbose: print(f" ⚠️  {jp.relative_to(src_path)} → unknown/skip")
                continue

            # data file
            try:
                df = _parse_dynamic_json(jp, fmt_hint=fmt_hint, r_scale_wtw3=r_scale_wtw3)
            except Exception as e:
                print(f" ❌  {jp} → parse error: {e}")
                continue

            (inst_dir / "requests.csv").write_text(df.to_csv(index=False), encoding="utf-8")

            # benchmark gần (nếu có)
            b = _find_benchmark(jp)
            has_bench = False
            if b:
                _write_benchmark_cleaned(b, inst_dir)
                has_bench = True

            # static (nếu có)
            s = _find_static_solution(src_path.parent if src_path.parent != src_path else repo_root, stem)
            if s:
                (inst_dir / f"static_solution{s.suffix}").write_text(
                    s.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8"
                )

            report.append((stem, len(df), has_bench))
            print(f"✅ {stem:>12s} → {len(df):4d} reqs | benchmark={has_bench}")

    if verbose:
        have_req = {p.parent.name for p in Path(dst_root).rglob("requests.csv")}
        have_bm  = {p.parent.name for p in Path(dst_root).rglob("benchmark.json")}
        miss_bm  = sorted(have_req - have_bm, key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)])
        if miss_bm:
            print(f"\nℹ️  Instances without benchmark: {', '.join(miss_bm[:20])}{' …' if len(miss_bm)>20 else ''}")

    print(f"\nDONE. {len(report)} instance(s) → {dst_root}")
    return report

# ---------- import static (no time window) ----------
def import_static_solutions(
    src_dir: str = "Static_solution_noTimeWindow",
    dst_root: str = "data/raw",
    horizon: float = 1e9,
    verbose: bool = False,
):
    src_path = Path(src_dir)
    dst = Path(dst_root); dst.mkdir(parents=True, exist_ok=True)
    if not src_path.exists():
        print(f"  Skip: '{src_dir}' not found."); return []

    files = _iter_source_files(src_path)
    out = []
    for p in files:
        m = re.match(r"(\d+\.\d+\.\d+)", p.name)
        stem = (m.group(1) if m else p.stem).replace(".json", "")
        inst_dir = dst / stem
        inst_dir.mkdir(parents=True, exist_ok=True)

        kind, obj = _classify_and_load(p)

        # benchmark/static solution → lưu lại để tham chiếu
        if kind == "benchmark":
            (inst_dir / "static_benchmark.json").write_text(
                json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if verbose: print(f"🗒  static benchmark for {stem}")
            out.append((stem, 0, True))
            continue

        # data tĩnh → tạo static_requests.csv
        if isinstance(obj, dict) and "requests" in obj:
            rows = obj["requests"]
        elif isinstance(obj, dict):
            df = _from_dict_of_arrays(obj)
            rows = None
        else:
            rows = obj

        if rows is not None:
            buf = []
            for i, r in enumerate(rows):
                if isinstance(r, dict):
                    x = r.get("x"); y = r.get("y")
                    demand = r.get("demand", r.get("demands"))
                else:
                    x, y, demand = r[:3]
                if x is None or y is None: continue
                buf.append([i, float(x), float(y), float(demand), 1, 0.0, 0.0, float(horizon)])
            df = pd.DataFrame(buf, columns=REQ_COLS)

        (inst_dir / "static_requests.csv").write_text(df.to_csv(index=False), encoding="utf-8")
        if verbose: print(f" {stem:>12s} → static_requests.csv ({len(df)} rows)")
        out.append((stem, len(df), False))
    return out

# ---------- static input (per-instance) ----------
def _load_static_input(cfg: dict) -> Optional[dict]:
    static_root = Path(cfg.get("static_input_root", "inputs_static"))
    inst = str(cfg["instance"])
    path = static_root / f"{inst}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _apply_static_input_to_cfg(cfg: dict, static_obj: dict):
    def _pick_depot(customers):
        if not customers:
            return None
        for c in customers:
            if c.get("index") == 0:
                return c
        return customers[0]

    custs = static_obj.get("customers") or []
    depot = _pick_depot(custs)
    if depot:
        cfg["depot"] = [float(depot.get("x", 0.0)), float(depot.get("y", 0.0))]

    veh = cfg.setdefault("vehicles", {})
    truck_info = static_obj.get("truck") or {}
    truck_count = static_obj.get("trucks_count")
    truck_speed = truck_info.get("V_max (m/s)") or truck_info.get("speed")
    truck_cap = truck_info.get("M_t (kg)") or truck_info.get("capacity")
    if truck_count is None or truck_speed is None or truck_cap is None:
        raise ValueError(f"Thieu thong tin truck trong static input cho instance {cfg['instance']}")
    veh["trucks"] = {
        "count": int(truck_count),
        "speed": float(truck_speed),
        "capacity": float(truck_cap),
    }

    drone_info = static_obj.get("drone") or {}
    drone_data = drone_info.get("_data", {}) if isinstance(drone_info, dict) else {}
    drone_count = static_obj.get("drones_count", 0)
    drone_speed = drone_data.get("V_max (m/s)") or drone_info.get("V_max (m/s)")
    drone_cap = drone_data.get("capacity [kg]") or drone_info.get("capacity")
    drone_radius = drone_info.get("radius") if isinstance(drone_info, dict) else None

    veh["drones"] = {
        "count": int(drone_count),
        "speed": float(drone_speed) if drone_speed is not None else 0.0,
        "capacity": float(drone_cap) if drone_cap is not None else 0.0,
    }
    if drone_radius is not None:
        veh["drones"]["radius"] = float(drone_radius)

# ---------- loader cho simulator ----------
def load_instance(cfg):
    root = Path(cfg["data_root"]); inst_dir = root / cfg["instance"]
    if not inst_dir.exists():
        raise FileNotFoundError(f"Không thấy thư mục instance: {inst_dir}")

    req = pd.read_csv(inst_dir / "requests.csv")
    for c in REQ_COLS:
        if c not in req.columns:
            if c == "req_id": req[c] = range(len(req))
            elif c == "drone_ok": req[c] = 1
            elif c == "t_arrive": req[c] = 0.0
            elif c == "e_i": req[c] = 0.0
            elif c == "l_i": req[c] = 1e9
            else: raise ValueError(f"Thiếu cột bắt buộc: {c}")
    req = req[REQ_COLS].copy()

    bench = None
    bj = inst_dir / "benchmark.json"
    if bj.exists():
        bench = json.loads(bj.read_text(encoding="utf-8"))
    else:
        br = inst_dir / "benchmark.raw.jsonc"
        if br.exists():
            bench = _load_json_relaxed(br)

    # Ap static input (vehicles/depot) neu co, de khong phai cau hinh chung trong config.yaml
    static_obj = _load_static_input(cfg)
    if static_obj:
        _apply_static_input_to_cfg(cfg, static_obj)
    elif "vehicles" not in cfg:
        raise ValueError(f"Thieu 'vehicles' trong cfg va khong tim thay static input cho {cfg['instance']}")

    # horizon mac dinh: lay max l_i neu chua thiet lap
    if "horizon" not in cfg:
        cfg["horizon"] = float(req["l_i"].max()) if len(req) else 1e9
    return req, bench

# ---------- main ----------
if __name__ == "__main__":
    verbose = ("--verbose" in sys.argv) or ("-v" in sys.argv)
    if "--static" in sys.argv:
        import_static_solutions(
            src_dir="Static_solution_noTimeWindow",
            dst_root="data/raw",
            horizon=1e9,
            verbose=verbose,
        )
    else:
        generate_data_from_dynamic_sources(
            src_dirs=("WithTimeWindows","WithTimeWindows3"),
            dst_root="data/raw",
            r_scale_wtw3=None,   # tắt scale r_i cho WTW3
            verbose=verbose,
        )
