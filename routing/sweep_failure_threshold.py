from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

DATA_PATH = Path(
    "benchmark/results/final_10k/"
    "e2b_all_10015_cases.csv"
)

MODEL_PATH = Path(
    "routing/e2b_calibrated_failure_router.joblib"
)

OUTPUT_PATH = Path(
    "benchmark/results/final_10k/"
    "calibrated_router_threshold_sweep.csv"
)

SEED = 40


def group_split(df):
    groups = df["image_id"].astype(str).values

    split1 = GroupShuffleSplit(
        n_splits=1,
        train_size=0.70,
        random_state=SEED,
    )

    train_idx, temp_idx = next(
        split1.split(df, groups=groups)
    )

    temp = df.iloc[temp_idx].reset_index(drop=True)

    temp_groups = temp["image_id"].astype(str).values

    split2 = GroupShuffleSplit(
        n_splits=1,
        train_size=0.50,
        random_state=SEED + 1,
    )

    val_idx, test_idx = next(
        split2.split(temp, groups=temp_groups)
    )

    test = temp.iloc[test_idx].reset_index(drop=True)

    return test


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(DATA_PATH)

df = df.dropna(
    subset=[
        "question",
        "image_id",
        "vqa_score",
    ]
).copy()

df["failure"] = (
    df["vqa_score"].astype(float) < 0.5
).astype(int)

test = group_split(df)

bundle = joblib.load(MODEL_PATH)

vectorizer = bundle["vectorizer"]
model = bundle["model"]

X_test = vectorizer.transform(
    test["question"].astype(str)
)

prob = model.predict_proba(X_test)[:, 1]

y = test["failure"].values


# ============================================================
# Global ranking quality
# ============================================================

print("=" * 100)
print("CALIBRATED ROUTER THRESHOLD SWEEP")
print("=" * 100)

print(f"Test cases          : {len(test):,}")
print(f"Actual failures     : {int(y.sum()):,}")
print(f"Actual successes    : {int((y == 0).sum()):,}")
print(f"ROC-AUC             : {roc_auc_score(y, prob):.4f}")
print(f"PR-AUC              : {average_precision_score(y, prob):.4f}")
print()


# ============================================================
# Thresholds
# ============================================================

thresholds = sorted(set(
    [0.10, 0.15, 0.20, 0.21, 0.25, 0.30, 0.35,
     0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
     0.75, 0.80]
))

rows = []

for threshold in thresholds:

    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    ).ravel()

    total = len(y)
    failures = tp + fn
    successes = tn + fp

    large_n = tp + fp
    small_n = tn + fn

    failure_recall = (
        tp / failures
        if failures else 0
    )

    failure_precision = (
        tp / large_n
        if large_n else 0
    )

    large_rate = (
        large_n / total
        if total else 0
    )

    over_escalation = (
        fp / successes
        if successes else 0
    )

    small_failure_rate = (
        fn / small_n
        if small_n else 0
    )

    small_success_rate = (
        tn / small_n
        if small_n else 0
    )

    rows.append({
        "threshold": threshold,
        "failure_recall": failure_recall,
        "failure_precision": failure_precision,
        "large_rate": large_rate,
        "small_rate": 1 - large_rate,
        "over_escalation": over_escalation,
        "small_failure_rate": small_failure_rate,
        "small_success_rate": small_success_rate,
        "failures_caught": tp,
        "failures_missed": fn,
        "successful_cases_escalated": fp,
        "successful_cases_kept_small": tn,
    })


result = pd.DataFrame(rows)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# Pretty print
# ============================================================

print(
    f"{'Thr':>5} "
    f"{'Catch':>8} "
    f"{'LARGE':>8} "
    f"{'OverEsc':>9} "
    f"{'SMALL Fail':>11} "
    f"{'SMALL Acc':>10} "
    f"{'Caught':>9} "
    f"{'Missed':>8}"
)

print("-" * 100)

for _, r in result.iterrows():

    print(
        f"{r['threshold']:>5.2f} "
        f"{r['failure_recall']*100:>7.1f}% "
        f"{r['large_rate']*100:>7.1f}% "
        f"{r['over_escalation']*100:>8.1f}% "
        f"{r['small_failure_rate']*100:>10.1f}% "
        f"{r['small_success_rate']*100:>9.1f}% "
        f"{int(r['failures_caught']):>9} "
        f"{int(r['failures_missed']):>8}"
    )


# ============================================================
# Useful operating points
# ============================================================

print()
print("=" * 100)
print("POSSIBLE OPERATING POINTS")
print("=" * 100)

for target in [0.80, 0.75, 0.70]:

    candidates = result[
        result["failure_recall"] >= target
    ]

    if len(candidates):

        best = candidates.sort_values(
            "large_rate"
        ).iloc[0]

        print(
            f"Catch >= {target*100:.0f}% failures: "
            f"threshold={best['threshold']:.2f}, "
            f"LARGE={best['large_rate']*100:.1f}%, "
            f"SMALL-failure={best['small_failure_rate']*100:.1f}%"
        )


print()
print("Saved:")
print(OUTPUT_PATH)
