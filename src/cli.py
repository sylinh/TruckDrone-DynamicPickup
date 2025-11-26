# src/cli.py
import argparse
from pathlib import Path
from datetime import datetime
import json, yaml, re, shutil

from .io_drive import load_instance
from .policy import build_baseline
from .sim import run_episode
from .gp import train_gphh, build_gp_policy_from_pair

# ---------- utils ----------
def load_yaml_utf8(path: str):
    p = Path(path)
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return yaml.safe_load(p.read_text(encoding="utf-8-sig"))

def _ts():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def _results_dir(cfg):
    root = Path(cfg.get("results_dir", "results"))
    return root / str(cfg["instance"])

def _bench_best(bench):
    if not isinstance(bench, dict):
        return None
    summ = bench.get("summary") or {}
    return summ.get("gp_best") or summ.get("heuristic") or None

def _service_ratio_from(d):
    if not isinstance(d, dict): return None
    for k in ("service_ratio","served_ratio"):
        if isinstance(d.get(k),(int,float)): return float(d[k])
    sv, tot = d.get("served"), d.get("total")
    if isinstance(sv,(int,float)) and isinstance(tot,(int,float)) and tot>0:
        return float(sv)/float(tot)
    return None

def _makespan_from(d):
    if not isinstance(d, dict): return None
    for k in ("makespan","T_finish","Tfinish"):
        if isinstance(d.get(k),(int,float)): return float(d[k])
    return None

def _compare_vs_benchmark(stats: dict, bench_best: dict):
    out = {"makespan": None, "service_ratio": None}
    ms_o = _makespan_from(stats); ms_b = _makespan_from(bench_best) if isinstance(bench_best, dict) else None
    if isinstance(ms_o,(int,float)) and isinstance(ms_b,(int,float)):
        out["makespan"] = {
            "ours": ms_o, "benchmark": ms_b,
            "abs_diff": ms_o - ms_b,
            "improvement_pct": (ms_b - ms_o)/ms_b*100.0 if ms_b!=0 else None
        }
    sr_o = _service_ratio_from(stats); sr_b = _service_ratio_from(bench_best) if isinstance(bench_best, dict) else None
    if isinstance(sr_o,(int,float)) and isinstance(sr_b,(int,float)):
        out["service_ratio"] = {
            "ours": sr_o, "benchmark": sr_b,
            "abs_diff": sr_o - sr_b,
            "improvement_pct": (sr_o - sr_b)/sr_b*100.0 if sr_b!=0 else None
        }
    return out

def _write_json(outdir: Path, base_name: str, payload: dict):
    outdir.mkdir(parents=True, exist_ok=True)
    ts = _ts()
    f_ts = outdir / f"{base_name}_{ts}.json"
    f_lt = outdir / f"{base_name}_latest.json"
    txt = json.dumps(payload, ensure_ascii=False, indent=2)
    f_ts.write_text(txt, encoding="utf-8")
    f_lt.write_text(txt, encoding="utf-8")
    print(f"💾 Saved: {f_ts}")

def _extract_pair_info(pair):
    info = {}
    try:
        info["routing"]    = getattr(pair, "routing_str", None) or getattr(pair, "routing", None)
        info["sequencing"] = getattr(pair, "sequencing_str", None) or getattr(pair, "sequencing", None)
    except Exception:
        pass
    if not info.get("routing") and isinstance(pair, (tuple,list)) and len(pair)>=1: info["routing"] = str(pair[0])
    if not info.get("sequencing") and isinstance(pair, (tuple,list)) and len(pair)>=2: info["sequencing"] = str(pair[1])
    if not info: info["raw"] = str(pair)
    return info

def _natsort_key_str(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def _list_instances(data_root: str, patterns: list[str] | None):
    root = Path(data_root)
    if not root.exists(): return []
    all_dirs = [p.name for p in root.iterdir() if p.is_dir()]
    if not patterns or patterns == ["all"]:
        return sorted(all_dirs, key=_natsort_key_str)
    # cho phép glob đơn giản: 6.*.*  |  10.10.*  |  6.5.1
    out = set()
    for pat in patterns:
        out |= set([d for d in all_dirs if Path(d).match(pat)])
    return sorted(out, key=_natsort_key_str)

def _clean_results_for_instance(cfg, inst: str):
    root = Path(cfg.get("results_dir", "results"))
    target = root / str(inst)
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)

# ---------- single run helpers ----------
def _run_eval_baseline(cfg_base: dict, instance: str, save_each: bool = True):
    cfg = dict(cfg_base); cfg["instance"] = instance
    req_df, bench = load_instance(cfg)
    pol = build_baseline()
    stats, _ = run_episode(cfg, pol, req_df)
    bench_best = _bench_best(bench)
    compare = _compare_vs_benchmark(stats, bench_best)
    if save_each:
        outdir = _results_dir(cfg)
        _write_json(outdir, "eval_result", {
            "mode":"baseline","instance":instance,"cfg":{"objective":cfg.get("objective"),"weights":cfg.get("weights")},
            "stats":stats,"benchmark_summary":(bench.get("summary") if isinstance(bench,dict) else None),
        })
        _write_json(outdir, "eval_vs_benchmark", {"mode":"baseline","instance":instance,"compare":compare})
    return stats, compare

def _run_train_eval_gp(cfg_base: dict, instance: str, save_each: bool = True):
    cfg = dict(cfg_base); cfg["instance"] = instance
    req_df, bench = load_instance(cfg)
    pair = train_gphh(cfg)
    pol = build_gp_policy_from_pair(cfg, pair)
    stats, _ = run_episode(cfg, pol, req_df)
    bench_best = _bench_best(bench)
    compare = _compare_vs_benchmark(stats, bench_best)
    if save_each:
        outdir = _results_dir(cfg)
        _write_json(outdir, "gp_train", {"mode":"gp_train","instance":instance,"gp_cfg":cfg.get("gp"),"best_pair":_extract_pair_info(pair)})
        _write_json(outdir, "gp_eval",  {"mode":"gp_eval","instance":instance,"cfg":{"objective":cfg.get("objective"),"weights":cfg.get("weights")},"stats":stats,"benchmark_summary":(bench.get("summary") if isinstance(bench,dict) else None)})
        _write_json(outdir, "gp_vs_benchmark", {"mode":"gp","instance":instance,"compare":compare})
    return stats, compare

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(prog="python -m src.cli")
    sub = ap.add_subparsers(dest="mode", required=True)

    sp_eval = sub.add_parser("eval", help="Evaluate baseline on single instance")
    sp_eval.add_argument("--config", default="config.yaml")
    sp_eval.add_argument("--clean-results", action="store_true", help="Delete existing results/<instance> before running")

    sp_train = sub.add_parser("train-gp", help="Train GP then evaluate on single instance")
    sp_train.add_argument("--config", default="config.yaml")
    sp_train.add_argument("--clean-results", action="store_true", help="Delete existing results/<instance> before running")

    sp_batch = sub.add_parser("batch", help="Run many instances and export summary table")
    sp_batch.add_argument("--config", default="config.yaml")
    sp_batch.add_argument("--run-mode", dest="run_mode", choices=["baseline","gp"], default="baseline",
                          help="What to run per-instance (default baseline)")
    sp_batch.add_argument("--instances", nargs="*", default=["all"],
                          help="List of instance ids or glob patterns (e.g., 6.*.* 10.10.*). Default: all")
    sp_batch.add_argument("--summary-only", action="store_true",
                          help="Only write batch summary (skip per-instance files)")
    sp_batch.add_argument("--out", default=None,
                          help="Override summary output path (CSV). Default: results/_batch/<mode>_summary_<ts>.csv")
    sp_batch.add_argument("--clean-results", action="store_true",
                          help="Delete existing results/<instance> before each run in the batch")

    args = ap.parse_args()
    cfg = load_yaml_utf8(getattr(args, "config", "config.yaml"))
    cfg.setdefault("data_root", "data/raw")
    cfg.setdefault("results_dir", "results")

    if ap.prog.endswith("eval") or args.mode == "eval":
        if "instance" not in cfg:
            ap.error("config.yaml thiếu khóa 'instance'")
        if getattr(args, "clean_results", False):
            _clean_results_for_instance(cfg, cfg["instance"])
        req_df, bench = load_instance(cfg)
        pol = build_baseline()
        stats, _ = run_episode(cfg, pol, req_df)
        print("Baseline:", stats)
        outdir = _results_dir(cfg)
        bench_best = _bench_best(bench)
        _write_json(outdir, "eval_result", {"mode":"baseline","instance":cfg["instance"],"cfg":{"objective":cfg.get("objective"),"weights":cfg.get("weights")},"stats":stats,"benchmark_summary":(bench.get("summary") if isinstance(bench,dict) else None)})
        _write_json(outdir, "eval_vs_benchmark", {"mode":"baseline","instance":cfg["instance"],"compare":_compare_vs_benchmark(stats, bench_best)})
        return

    if args.mode == "train-gp":
        if "instance" not in cfg:
            ap.error("config.yaml thi?u kh?a 'instance'")
        if getattr(args, "clean_results", False):
            _clean_results_for_instance(cfg, cfg["instance"])
        req_df, bench = load_instance(cfg)
        pair = train_gphh(cfg)
        pol = build_gp_policy_from_pair(cfg, pair)
        stats, _ = run_episode(cfg, pol, req_df)
        print("GPPolicy:", stats)
        stats, _ = run_episode(cfg, pol, req_df)
        print("GPPolicy:", stats)
        outdir = _results_dir(cfg)
        bench_best = _bench_best(bench)
        _write_json(outdir, "gp_train", {"mode":"gp_train","instance":cfg["instance"],"gp_cfg":cfg.get("gp"),"best_pair":_extract_pair_info(pair)})
        _write_json(outdir, "gp_eval",  {"mode":"gp_eval","instance":cfg["instance"],"cfg":{"objective":cfg.get("objective"),"weights":cfg.get("weights")},"stats":stats,"benchmark_summary":(bench.get("summary") if isinstance(bench,dict) else None)})
        _write_json(outdir, "gp_vs_benchmark", {"mode":"gp","instance":cfg["instance"],"compare":_compare_vs_benchmark(stats, bench_best)})
        return

    # ---------- batch ----------
    if args.mode == "batch":
        # xác định danh sách instances
        insts = _list_instances(cfg["data_root"], args.instances)
        if not insts:
            print("⚠️  Không tìm thấy instance nào."); return

        rows = []
        for inst in insts:
            print(f"▶ {args.run_mode} on {inst} ...")
            if getattr(args, "clean_results", False):
                _clean_results_for_instance(cfg, inst)
            if args.run_mode == "baseline":
                stats, cmpres = _run_eval_baseline(cfg, inst, save_each=(not args.summary_only))
            else:  # gp
                stats, cmpres = _run_train_eval_gp(cfg, inst, save_each=(not args.summary_only))

            ms = cmpres.get("makespan") or {}
            sr = cmpres.get("service_ratio") or {}
            row = {
                "instance": inst,
                "makespan_ours": ms.get("ours"),
                "makespan_benchmark": ms.get("benchmark"),
                "makespan_abs_diff": ms.get("abs_diff"),
                "makespan_impr_%": ms.get("improvement_pct"),
                "service_ratio_ours": sr.get("ours"),
                "service_ratio_benchmark": sr.get("benchmark"),
                "service_ratio_abs_diff": sr.get("abs_diff"),
                "service_ratio_impr_%": sr.get("improvement_pct"),
            }
            rows.append(row)

        # ghi summary CSV + JSON
        batch_dir = Path(cfg.get("results_dir","results")) / "_batch"
        batch_dir.mkdir(parents=True, exist_ok=True)
        base = f"{args.run_mode}_summary"
        ts = _ts()
        csv_path = Path(args.out) if args.out else (batch_dir / f"{base}_{ts}.csv")
        json_path = batch_dir / f"{base}_{ts}.json"
        # CSV
        import csv
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        (batch_dir / f"{base}_latest.csv").write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
        # JSON
        payload = {"mode": args.run_mode, "count": len(rows), "rows": rows}
        json_txt = json.dumps(payload, ensure_ascii=False, indent=2)
        json_path.write_text(json_txt, encoding="utf-8")
        (batch_dir / f"{base}_latest.json").write_text(json_txt, encoding="utf-8")

        print(f"Summary saved:\n - {csv_path}\n - {json_path}")
        return

if __name__ == "__main__":
    main()
