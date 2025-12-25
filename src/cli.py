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
    print(f"Saved: {f_ts}")

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
    return stats, compare, bench_best

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
    return stats, compare, bench_best

def _run_eval_ml(cfg_base: dict, instance: str, model_path: str, log_path: str | None = None, save_each: bool = True):
    from .policy import MLPolicy
    from .ml_logging import DecisionLogger
    cfg = dict(cfg_base); cfg["instance"] = instance
    req_df, bench = load_instance(cfg)
    logger = DecisionLogger(log_path) if log_path else None
    pol = MLPolicy(model_path=model_path, logger=logger, instance=instance)
    stats, _ = run_episode(cfg, pol, req_df)
    if logger: logger.close()
    bench_best = _bench_best(bench)
    compare = _compare_vs_benchmark(stats, bench_best)
    if save_each:
        outdir = _results_dir(cfg)
        _write_json(outdir, "ml_eval",  {"mode":"ml","instance":instance,"cfg":{"objective":cfg.get("objective"),"weights":cfg.get("weights")},"stats":stats,"benchmark_summary":(bench.get("summary") if isinstance(bench,dict) else None),"model_path":model_path})
        _write_json(outdir, "ml_vs_benchmark", {"mode":"ml","instance":instance,"compare":compare})
    return stats, compare, bench_best

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(prog="python -m src.cli")
    sub = ap.add_subparsers(dest="mode", required=True)

    sp_eval = sub.add_parser("eval", help="Evaluate baseline on single instance")
    sp_eval.add_argument("--config", default="config.yaml")
    sp_eval.add_argument("--clean-results", action="store_true", help="Delete existing results/<instance> before running")

    sp_eval_ml = sub.add_parser("eval-ml", help="Evaluate ML policy on single instance")
    sp_eval_ml.add_argument("--config", default="config.yaml")
    sp_eval_ml.add_argument("--model", required=True, help="Path to ML model (.pkl)")
    sp_eval_ml.add_argument("--ml-log", default=None, help="Optional CSV to log decisions/features")
    sp_eval_ml.add_argument("--clean-results", action="store_true", help="Delete existing results/<instance> before running")

    sp_train = sub.add_parser("train-gp", help="Train GP then evaluate on single instance")
    sp_train.add_argument("--config", default="config.yaml")
    sp_train.add_argument("--clean-results", action="store_true", help="Delete existing results/<instance> before running")

    sp_batch = sub.add_parser("batch", help="Run many instances and export summary table")
    sp_batch.add_argument("--config", default="config.yaml")
    sp_batch.add_argument("--run-mode", dest="run_mode", choices=["baseline","gp","ml"], default="baseline",
                          help="What to run per-instance (default baseline)")
    sp_batch.add_argument("--instances", nargs="*", default=["all"],
                          help="List of instance ids or glob patterns (e.g., 6.*.* 10.10.*). Default: all")
    sp_batch.add_argument("--summary-only", action="store_true",
                          help="Only write batch summary (skip per-instance files)")
    sp_batch.add_argument("--out", default=None,
                          help="Override summary output path (CSV). Default: results/_batch/<mode>_summary_<ts>.csv")
    sp_batch.add_argument("--clean-results", action="store_true",
                          help="Delete existing results/<instance> before each run in the batch")
    sp_batch.add_argument("--model", default=None, help="Path to ML model (.pkl) when run-mode=ml")
    sp_batch.add_argument("--ml-log", default=None, help="Optional CSV to log decisions when run-mode=ml")

    args = ap.parse_args()
    cfg = load_yaml_utf8(getattr(args, "config", "config.yaml"))
    cfg.setdefault("data_root", "data/raw")
    cfg.setdefault("results_dir", "results")

    if ap.prog.endswith("eval") or args.mode == "eval":
        if "instance" not in cfg:
            ap.error("config.yaml missing key 'instance'")
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
            ap.error("config.yaml missing key 'instance'")
        if getattr(args, "clean_results", False):
            _clean_results_for_instance(cfg, cfg["instance"])
        req_df, bench = load_instance(cfg)
        pair = train_gphh(cfg)
        pol = build_gp_policy_from_pair(cfg, pair)
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
        insts = _list_instances(cfg["data_root"], args.instances)
        if not insts:
            print("Warning: no instances found."); return

        rows = []
        compare_rows = []
        for inst in insts:
            print(f"[batch] {args.run_mode} on {inst} ...")
            if getattr(args, "clean_results", False):
                _clean_results_for_instance(cfg, inst)
            if args.run_mode == "baseline":
                stats, cmpres, bench_best = _run_eval_baseline(cfg, inst, save_each=(not args.summary_only))
            elif args.run_mode == "gp":
                stats, cmpres, bench_best = _run_train_eval_gp(cfg, inst, save_each=(not args.summary_only))
            else:  # ml
                model_path = args.model or cfg.get("ml", {}).get("model_path")
                if not model_path:
                    ap.error("Missing --model (or ml.model_path in config) for run-mode=ml")
                log_path = args.ml_log
                stats, cmpres, bench_best = _run_eval_ml(cfg, inst, model_path=model_path, log_path=log_path, save_each=(not args.summary_only))

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

            compare_rows.append({
                "instance": inst,
                "served_ours": stats.get("served"),
                "total": stats.get("total"),
                "dropped_ours": stats.get("dropped"),
                "served_benchmark": (bench_best.get("served") if bench_best else None),
                "dropped_benchmark": (None if bench_best is None else (None if bench_best.get("served") is None else (stats.get("total") - bench_best.get("served") if stats and stats.get("total") is not None else None))),
                "makespan_ours": stats.get("makespan"),
                "makespan_benchmark": (bench_best.get("makespan") if bench_best else None),
                "makespan_impr_%": ms.get("improvement_pct"),
                "service_ratio_impr_%": sr.get("improvement_pct"),
            })

        # summary CSV + JSON
        batch_dir = Path(cfg.get("results_dir","results")) / "_batch"
        batch_dir.mkdir(parents=True, exist_ok=True)
        base = f"{args.run_mode}_summary"
        ts = _ts()
        csv_path = Path(args.out) if args.out else (batch_dir / f"{base}_{ts}.csv")
        json_path = batch_dir / f"{base}_{ts}.json"
        import csv
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        (batch_dir / f"{base}_latest.csv").write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
        payload = {"mode": args.run_mode, "count": len(rows), "rows": rows}
        json_txt = json.dumps(payload, ensure_ascii=False, indent=2)
        json_path.write_text(json_txt, encoding="utf-8")
        (batch_dir / f"{base}_latest.json").write_text(json_txt, encoding="utf-8")

        print(f"Summary saved:\n - {csv_path}\n - {json_path}")
        # compare folder (served/total/dropped vs benchmark), clean each run
        compare_dir = Path(cfg.get("results_dir","results")) / "_compare"
        shutil.rmtree(compare_dir, ignore_errors=True)
        compare_dir.mkdir(parents=True, exist_ok=True)
        cmp_base = f"{args.run_mode}_compare"
        cmp_csv = compare_dir / f"{cmp_base}_{ts}.csv"
        cmp_json = compare_dir / f"{cmp_base}_{ts}.json"
        import csv
        with cmp_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(compare_rows[0].keys()))
            w.writeheader(); w.writerows(compare_rows)
        cmp_payload = {"mode": args.run_mode, "count": len(compare_rows), "rows": compare_rows}
        cmp_txt = json.dumps(cmp_payload, ensure_ascii=False, indent=2)
        cmp_json.write_text(cmp_txt, encoding="utf-8")
        (compare_dir / f"{cmp_base}_latest.csv").write_text(cmp_csv.read_text(encoding="utf-8"), encoding="utf-8")
        (compare_dir / f"{cmp_base}_latest.json").write_text(cmp_txt, encoding="utf-8")
        print(f"Compare saved:\n - {cmp_csv}\n - {cmp_json}")
        return

if __name__ == "__main__":
    main()
