from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)


SEED = 40

OLD_DATA = Path(
    "benchmark/results/final_10k/"
    "e2b_all_10015_cases.csv"
)

NEW_DATA = Path(
    "benchmark/results/vqav2_e2b_30k.csv"
)

OLD_MODEL = Path(
    "routing/e2b_calibrated_failure_router.joblib"
)

OUTPUT_MODEL = Path(
    "routing/e2b_calibrated_failure_router_30k.joblib"
)


# ============================================================
# Reproduce EXACT old 10k val/test split
# ============================================================

old = pd.read_csv(OLD_DATA)

old = old.dropna(
    subset=["question", "image_id", "vqa_score"]
).copy()

old["failure"] = (
    old["vqa_score"].astype(float) < 0.5
).astype(int)

groups = old["image_id"].astype(str).values

split1 = GroupShuffleSplit(
    n_splits=1,
    train_size=0.70,
    random_state=SEED,
)

old_train_idx, temp_idx = next(
    split1.split(old, groups=groups)
)

temp = old.iloc[temp_idx].reset_index(drop=True)

temp_groups = temp["image_id"].astype(str).values

split2 = GroupShuffleSplit(
    n_splits=1,
    train_size=0.50,
    random_state=SEED + 1,
)

val_idx, test_idx = next(
    split2.split(temp, groups=temp_groups)
)

val = temp.iloc[val_idx].reset_index(drop=True)
test = temp.iloc[test_idx].reset_index(drop=True)

val_image_ids = set(
    val["image_id"].astype(str)
)

test_image_ids = set(
    test["image_id"].astype(str)
)

heldout_image_ids = (
    val_image_ids | test_image_ids
)

print("=" * 80)
print("FIXED HELD-OUT SETS")
print("=" * 80)

print("Validation:", len(val))
print("Test      :", len(test))
print()


# ============================================================
# Load full 30k
# ============================================================

df = pd.read_csv(NEW_DATA)

df = df.dropna(
    subset=["question", "image_id", "vqa_score"]
).copy()

df["failure"] = (
    df["vqa_score"].astype(float) < 0.5
).astype(int)

# IMPORTANT:
# remove every image appearing in old validation/test.
#
# This prevents leakage even if the 30k dataset contains
# additional questions for those same images.
train = df[
    ~df["image_id"]
    .astype(str)
    .isin(heldout_image_ids)
].reset_index(drop=True)

print("=" * 80)
print("30K TRAINING DATA")
print("=" * 80)

print(f"Available 30k cases : {len(df):,}")
print(f"Training cases      : {len(train):,}")
print(f"Failure rate        : {train.failure.mean():.3f}")
print()


# ============================================================
# Feature extractor
# ============================================================

vectorizer = FeatureUnion([
    (
        "word",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=3,
            max_features=40000,
            sublinear_tf=True,
            strip_accents="unicode",
        ),
    ),
    (
        "char",
        TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 4),
            min_df=4,
            max_features=30000,
            sublinear_tf=True,
        ),
    ),
])

print("Building TF-IDF...")

X_train = vectorizer.fit_transform(
    train["question"].astype(str)
)

X_val = vectorizer.transform(
    val["question"].astype(str)
)

X_test = vectorizer.transform(
    test["question"].astype(str)
)

y_train = train["failure"].values
y_val = val["failure"].values
y_test = test["failure"].values


# ============================================================
# Calibrated model
# ============================================================

base_model = LogisticRegression(
    C=0.3,
    max_iter=2500,
    solver="liblinear",
    class_weight=None,
    random_state=SEED,
)

model = CalibratedClassifierCV(
    estimator=base_model,
    method="sigmoid",
    cv=5,
)

print("Training 30k calibrated router...")

model.fit(
    X_train,
    y_train,
)

print("Training complete.")
print()


# ============================================================
# Probabilities
# ============================================================

val_prob = model.predict_proba(
    X_val
)[:, 1]

test_prob = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# Threshold sweep on VALIDATION ONLY
# ============================================================

rows = []

for threshold in np.arange(
    0.10,
    0.501,
    0.01,
):
    pred = (
        val_prob >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        pred,
        labels=[0, 1],
    ).ravel()

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0
    )

    large_rate = pred.mean()

    small_n = tn + fn

    small_failure_rate = (
        fn / small_n
        if small_n
        else 0
    )

    rows.append(
        (
            threshold,
            recall,
            large_rate,
            small_failure_rate,
        )
    )


# Keep same operating philosophy:
# target ~75% failure recall, then minimize LARGE usage.
valid = [
    x for x in rows
    if x[1] >= 0.75
]

if valid:
    chosen = sorted(
        valid,
        key=lambda x: x[2]
    )[0]

    threshold = float(
        chosen[0]
    )
else:
    threshold = 0.25


# ============================================================
# Evaluation helper
# ============================================================

def report(name, y, prob, threshold):

    pred = (
        prob >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    ).ravel()

    small_n = tn + fn

    print("=" * 80)
    print(name)
    print("=" * 80)

    print(
        f"ROC-AUC              : "
        f"{roc_auc_score(y, prob):.4f}"
    )

    print(
        f"PR-AUC               : "
        f"{average_precision_score(y, prob):.4f}"
    )

    print(
        f"Brier                : "
        f"{brier_score_loss(y, prob):.4f}"
    )

    print(
        f"Mean predicted risk  : "
        f"{prob.mean():.4f}"
    )

    print(
        f"Actual failure rate  : "
        f"{y.mean():.4f}"
    )

    print(
        f"Threshold            : "
        f"{threshold:.3f}"
    )

    print(
        f"Failure recall       : "
        f"{tp/(tp+fn):.4f}"
    )

    print(
        f"LARGE rate           : "
        f"{pred.mean():.4f}"
    )

    print(
        f"SMALL failure rate   : "
        f"{fn/small_n:.4f}"
        if small_n
        else "SMALL failure rate   : N/A"
    )

    print(
        f"Caught / failures    : "
        f"{tp}/{tp+fn}"
    )

    print()


report(
    "30K VALIDATION",
    y_val,
    val_prob,
    threshold,
)

report(
    "30K FIXED HELD-OUT TEST",
    y_test,
    test_prob,
    threshold,
)


# ============================================================
# Compare against old 10k calibrated model
# ============================================================

old_bundle = joblib.load(
    OLD_MODEL
)

old_X_test = (
    old_bundle["vectorizer"]
    .transform(
        test["question"].astype(str)
    )
)

old_prob = (
    old_bundle["model"]
    .predict_proba(old_X_test)[:, 1]
)

print("=" * 80)
print("10K vs 30K — SAME FIXED TEST SET")
print("=" * 80)

print(
    f"10k ROC-AUC : "
    f"{roc_auc_score(y_test, old_prob):.4f}"
)

print(
    f"30k ROC-AUC : "
    f"{roc_auc_score(y_test, test_prob):.4f}"
)

print()

print(
    f"10k PR-AUC  : "
    f"{average_precision_score(y_test, old_prob):.4f}"
)

print(
    f"30k PR-AUC  : "
    f"{average_precision_score(y_test, test_prob):.4f}"
)

print()

print(
    f"10k Brier   : "
    f"{brier_score_loss(y_test, old_prob):.4f}"
)

print(
    f"30k Brier   : "
    f"{brier_score_loss(y_test, test_prob):.4f}"
)


# ============================================================
# Wearable-style sanity check
# ============================================================

questions = [
    "What is in front of me?",
    "What am I looking at?",
    "How many objects are on the table?",
    "Why is the person doing that?",
    "What color is the cup?",
    "Where is the phone?",
    "Is there a laptop in front of me?",
    "What is this person doing?",
    "What happened before the person sat down?",
]

X_live = vectorizer.transform(
    questions
)

live_prob = model.predict_proba(
    X_live
)[:, 1]

print()
print("=" * 80)
print("30K LIVE-STYLE SANITY CHECK")
print("=" * 80)

for q, p in zip(
    questions,
    live_prob,
):
    route = (
        "LARGE"
        if p >= threshold
        else "SMALL"
    )

    print(
        f"{route:<5} "
        f"risk={p:.3f}  "
        f"{q}"
    )


# ============================================================
# Save
# ============================================================

bundle = {
    "vectorizer": vectorizer,
    "model": model,
    "threshold": threshold,
    "global_failure_rate":
        float(train.failure.mean()),
    "failure_definition":
        "vqa_score < 0.5",
    "calibrated": True,
    "training_source":
        "VQAv2 30k E2B benchmark",
    "fixed_10k_holdout":
        True,
}

joblib.dump(
    bundle,
    OUTPUT_MODEL,
)

print()
print("Saved:")
print(OUTPUT_MODEL)
