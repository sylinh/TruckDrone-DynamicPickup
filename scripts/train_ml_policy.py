"""
Train ML models for routing/sequencing from logged CSV (see src/ml_logging.py).

Usage:
  python scripts/train_ml_policy.py --data results/ml_logs.csv --out models/ml_policy.pkl
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

DECISION_TYPES = ["route", "sequence"]


def train_models(df: pd.DataFrame):
    models = {}
    feature_cols = [c for c in df.columns if c not in {"decision_type","instance","time","veh_idx","req_idx","chosen","score"}]
    for dt in DECISION_TYPES:
        sub = df[df["decision_type"] == dt]
        if sub.empty:
            continue
        X = sub[feature_cols]
        y = sub["chosen"].astype(int)
        clf = GradientBoostingClassifier(random_state=42)
        clf.fit(X, y)
        models[dt] = clf
    return models, feature_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV logs from DecisionLogger")
    ap.add_argument("--out", required=True, help="Output pickle path")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    models, feature_cols = train_models(df)
    if not models:
        print("No models trained (no data)"); return
    payload = {"models": models, "features": feature_cols}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"Saved ML policy to {out_path}")


if __name__ == "__main__":
    main()
