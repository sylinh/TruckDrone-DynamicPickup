# Dynamic Truck-Drone Pickup Scheduling (GP-based)

A fleet of trucks and drones serves dynamic pickup requests with time windows. We evolve routing (R) and sequencing (S) rules via Genetic Programming and simulate with an event-driven dispatcher (per-vehicle queues, hard TW/capacity/Lw checks).

## Quick start

```bash
pip install -r requirements.txt

# baseline eval 1 instance (uses config.yaml -> instance, vehicles, constraints.Lw, objective.lambda_w, gp.*)
python -m src.cli eval --config config.yaml --clean-results

# train GP then eval 1 instance
python -m src.cli train-gp --config config.yaml --clean-results

# batch many instances (baseline or gp)
python -m src.cli batch --run-mode baseline --instances "6.*.*" --config config.yaml --clean-results
# or GP: --run-mode gp
```

Results are written to `results/<instance>/`; batch summary to `results/_batch/`. Use `--clean-results` to wipe old outputs before each run.

## Scripts

- Plot benchmark comparisons: `python scripts/plot_results.py --results-dir results --out-dir results/_plots`
- Sweep Lw values and plot: `python scripts/tune_lw.py --config config.yaml --instances 6.5.1 6.5.2 --lw 180 240 300 360 480 720 1e9 --out results/_lw`

## Model highlights

- Features (shared for R/S): dist_norm, demand_norm, rem_capacity_norm, demand_over_rc, nearest_next_norm, time_to_ready, time_to_due, waiting_time, drone_ok, is_drone (normalized by D_max/H/Q_max).
- Fitness: `lambda_w * (makespan / T_ref) + (1 - service_ratio)`, with T_ref from baseline makespan.
- Simulator: event-driven, per-vehicle queues; request assignment via R; next-customer selection via S; drops queue when no feasible choice to avoid deadlock.
