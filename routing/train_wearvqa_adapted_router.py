from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]

WEAR_PATH = (
    ROOT
    / "routing/benchmarks/results/"
    / "wearvqa_e2b_image_graded.csv"
)

VQA_PATH = (
    ROOT
    / "benchmark/results/"
    / "vqav2_e2b_30k.csv"
)

OUT_MODEL = (
    ROOT
    / "routing/"
    / "e2b_wearvqa_adapted_failure_router.joblib"
)

OUT_SPLIT = (
    ROOT
    / "routing/benchmarks/results/"
    / "wearvqa_router_split.csv"
)

OUT_SUMMARY = (
    ROOT
    / "routing/benchmarks/results/"
    / "wearvqa_router_adaptation_summary.csv"
)

RANDOM_STATE = 42

# Operational target:
# catch at least this fraction of SMALL failures on validation.
TARGET_FAILURE_RECALL = 0.80

WEAR_WEIGHTS = [
    1.0,
    2.0,
    3.0,
    5.0,
    10.0,
]


# ============================================================
# Load WearVQA
# ============================================================

wear = pd.read_csv(WEAR_PATH)

wear = wear[
    wear["small_failure"].notna()
].copy()

wear["small_failure"] = (
    wear["small_failure"]
    .astype(int)
)

wear["question"] = (
    wear["question"]
    .fillna("")
    .astype(str)
)

print("=" * 90)
print("WEARVQA DOMAIN ADAPTATION")
print("=" * 90)

print(
    f"WearVQA judgeable : {len(wear):,}"
)

print(
    f"Wear failure rate : "
    f"{wear['small_failure'].mean()*100:.2f}%"
)


# ============================================================
# Group split WearVQA: 70 / 15 / 15
#
# Group by image if available so questions from the same image
# cannot leak between train and evaluation.
# ============================================================

if "image" in wear.columns:
    groups = wear["image"].astype(str)
else:
    groups = wear["sample_id"].astype(str)


gss1 = GroupShuffleSplit(
    n_splits=1,
    train_size=0.70,
    random_state=RANDOM_STATE,
)

train_idx, rest_idx = next(
    gss1.split(
        wear,
        wear["small_failure"],
        groups,
    )
)

wear_train = wear.iloc[
    train_idx
].copy()

wear_rest = wear.iloc[
    rest_idx
].copy()


if "image" in wear_rest.columns:
    rest_groups = (
        wear_rest["image"]
        .astype(str)
    )
else:
    rest_groups = (
        wear_rest["sample_id"]
        .astype(str)
    )


gss2 = GroupShuffleSplit(
    n_splits=1,
    train_size=0.50,
    random_state=RANDOM_STATE + 1,
)

val_rel, test_rel = next(
    gss2.split(
        wear_rest,
        wear_rest["small_failure"],
        rest_groups,
    )
)

wear_val = wear_rest.iloc[
    val_rel
].copy()

wear_test = wear_rest.iloc[
    test_rel
].copy()


print()
print("WearVQA split:")

for name, x in [
    ("train", wear_train),
    ("val", wear_val),
    ("test", wear_test),
]:
    print(
        f"{name:<5}: "
        f"{len(x):4d}  "
        f"failure={x['small_failure'].mean()*100:5.1f}%"
    )


# Save fixed split so every future experiment uses the same test set.
wear_split = wear.copy()
wear_split["split"] = "unused"

wear_split.loc[
    wear_train.index,
    "split",
] = "train"

wear_split.loc[
    wear_val.index,
    "split",
] = "val"

wear_split.loc[
    wear_test.index,
    "split",
] = "test"

wear_split.to_csv(
    OUT_SPLIT,
    index=False,
)


# ============================================================
# Load VQAv2 generic labels
# ============================================================

vqa = pd.read_csv(VQA_PATH)

vqa = vqa[
    vqa["vqa_score"].notna()
].copy()

vqa["question"] = (
    vqa["question"]
    .fillna("")
    .astype(str)
)

vqa["small_failure"] = (
    vqa["vqa_score"] < 0.5
).astype(int)

print()
print(
    f"VQAv2 available   : {len(vqa):,}"
)

print(
    f"VQAv2 failure rate: "
    f"{vqa['small_failure'].mean()*100:.2f}%"
)


# ============================================================
# Feature extraction
# ============================================================

vectorizer = FeatureUnion([
    (
        "word",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            max_features=100000,
        ),
    ),
    (
        "char",
        TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 4),
            min_df=2,
            sublinear_tf=True,
            max_features=100000,
        ),
    ),
])


# Fit vocabulary using ONLY training-side text.
all_training_text = pd.concat([
    vqa["question"],
    wear_train["question"],
]).tolist()

print()
print("Fitting TF-IDF...")

vectorizer.fit(
    all_training_text
)


X_vqa = vectorizer.transform(
    vqa["question"]
)

y_vqa = (
    vqa["small_failure"]
    .to_numpy()
)


X_wear_train = vectorizer.transform(
    wear_train["question"]
)

y_wear_train = (
    wear_train["small_failure"]
    .to_numpy()
)


X_val = vectorizer.transform(
    wear_val["question"]
)

y_val = (
    wear_val["small_failure"]
    .to_numpy()
)


X_test = vectorizer.transform(
    wear_test["question"]
)

y_test = (
    wear_test["small_failure"]
    .to_numpy()
)


# ============================================================
# Helpers
# ============================================================

def fit_model(X, y, sample_weight=None):

    base = LogisticRegression(
        max_iter=2000,
        C=1.0,
        solver="liblinear",
    )

    model = CalibratedClassifierCV(
        base,
        method="sigmoid",
        cv=5,
    )

    if sample_weight is None:
        model.fit(
            X,
            y,
        )
    else:
        model.fit(
            X,
            y,
            sample_weight=sample_weight,
        )

    return model


def choose_threshold(
    y_true,
    prob,
    target_recall=0.80,
):

    best = None

    # Highest threshold that still reaches the desired
    # failure recall = least cloud escalation.
    for threshold in np.arange(
        0.01,
        1.00,
        0.01,
    ):

        pred_large = (
            prob >= threshold
        )

        failures = (
            y_true == 1
        )

        if failures.sum() == 0:
            continue

        recall = (
            pred_large[failures]
            .mean()
        )

        large_rate = (
            pred_large.mean()
        )

        if recall >= target_recall:

            candidate = {
                "threshold": threshold,
                "recall": recall,
                "large_rate": large_rate,
            }

            if (
                best is None
                or threshold > best["threshold"]
            ):
                best = candidate

    return best


def evaluate(
    name,
    model,
    threshold,
):

    prob = model.predict_proba(
        X_test
    )[:, 1]

    large = (
        prob >= threshold
    )

    failures = (
        y_test == 1
    )

    successes = (
        y_test == 0
    )

    caught = (
        large[failures].mean()
        if failures.sum()
        else np.nan
    )

    large_rate = (
        large.mean()
    )

    over_escalation = (
        large[successes].mean()
        if successes.sum()
        else np.nan
    )

    kept_small = ~large

    small_failure_rate = (
        y_test[kept_small].mean()
        if kept_small.sum()
        else np.nan
    )

    return {
        "model": name,
        "test_auc": roc_auc_score(
            y_test,
            prob,
        ),
        "test_pr_auc": average_precision_score(
            y_test,
            prob,
        ),
        "test_brier": brier_score_loss(
            y_test,
            prob,
        ),
        "threshold": threshold,
        "failure_recall": caught,
        "large_rate": large_rate,
        "over_escalation": over_escalation,
        "small_failure_rate": small_failure_rate,
    }


# ============================================================
# Experiment A: VQAv2 only
# ============================================================

print()
print("=" * 90)
print("A. VQAv2 ONLY")
print("=" * 90)

model_vqa = fit_model(
    X_vqa,
    y_vqa,
)

val_prob = model_vqa.predict_proba(
    X_val
)[:, 1]

choice = choose_threshold(
    y_val,
    val_prob,
    TARGET_FAILURE_RECALL,
)

result_vqa = evaluate(
    "VQAv2-only",
    model_vqa,
    choice["threshold"],
)

print(result_vqa)


# ============================================================
# Experiment B: WearVQA only
# ============================================================

print()
print("=" * 90)
print("B. WearVQA ONLY")
print("=" * 90)

model_wear = fit_model(
    X_wear_train,
    y_wear_train,
)

val_prob = model_wear.predict_proba(
    X_val
)[:, 1]

choice = choose_threshold(
    y_val,
    val_prob,
    TARGET_FAILURE_RECALL,
)

result_wear = evaluate(
    "WearVQA-only",
    model_wear,
    choice["threshold"],
)

print(result_wear)


# ============================================================
# Experiment C: VQAv2 + weighted WearVQA
# ============================================================

from scipy.sparse import vstack


X_combined = vstack([
    X_vqa,
    X_wear_train,
])

y_combined = np.concatenate([
    y_vqa,
    y_wear_train,
])


adaptation_results = []
models = {}


print()
print("=" * 90)
print("C. VQAv2 + WEARVQA")
print("=" * 90)


for weight in WEAR_WEIGHTS:

    print()
    print(
        f"WearVQA weight = {weight}"
    )

    weights = np.concatenate([
        np.ones(
            len(y_vqa),
            dtype=float,
        ),
        np.full(
            len(y_wear_train),
            weight,
            dtype=float,
        ),
    ])

    model = fit_model(
        X_combined,
        y_combined,
        sample_weight=weights,
    )

    val_prob = model.predict_proba(
        X_val
    )[:, 1]

    choice = choose_threshold(
        y_val,
        val_prob,
        TARGET_FAILURE_RECALL,
    )

    if choice is None:
        print(
            "Could not reach target recall."
        )
        continue

    # Selection criterion:
    # among models reaching target recall,
    # minimize LARGE calls on WearVQA validation.
    record = {
        "weight": weight,
        "val_threshold": choice["threshold"],
        "val_failure_recall": choice["recall"],
        "val_large_rate": choice["large_rate"],
    }

    adaptation_results.append(
        record
    )

    models[weight] = model

    print(record)


# Best = lowest cloud escalation while maintaining recall.
best = sorted(
    adaptation_results,
    key=lambda x: (
        x["val_large_rate"],
        -x["val_failure_recall"],
    ),
)[0]

best_weight = best["weight"]
best_threshold = best["val_threshold"]
best_model = models[best_weight]


result_adapted = evaluate(
    f"VQAv2+WearVQA(w={best_weight:g})",
    best_model,
    best_threshold,
)


# ============================================================
# Final comparison on untouched WearVQA test
# ============================================================

results = pd.DataFrame([
    result_vqa,
    result_wear,
    result_adapted,
])


print()
print("=" * 90)
print("FINAL HELD-OUT WEARVQA TEST")
print("=" * 90)

show = results.copy()

for col in [
    "test_auc",
    "test_pr_auc",
    "test_brier",
    "failure_recall",
    "large_rate",
    "over_escalation",
    "small_failure_rate",
]:

    show[col] = (
        show[col]
        .map(
            lambda x: (
                f"{x:.4f}"
                if pd.notna(x)
                else "nan"
            )
        )
    )

print(
    show.to_string(
        index=False
    )
)


# ============================================================
# Save best adapted model
# ============================================================

artifact = {
    "vectorizer": vectorizer,
    "model": best_model,
    "threshold": float(
        best_threshold
    ),
    "wear_weight": float(
        best_weight
    ),
    "target_failure_recall": float(
        TARGET_FAILURE_RECALL
    ),
    "label_definition": {
        "CORRECT": 0,
        "PARTIAL": 0,
        "INCORRECT": 1,
        "AMBIGUOUS": "excluded",
    },
}

joblib.dump(
    artifact,
    OUT_MODEL,
)

results.to_csv(
    OUT_SUMMARY,
    index=False,
)


print()
print("=" * 90)
print("SELECTED ADAPTED ROUTER")
print("=" * 90)

print(
    f"WearVQA weight : {best_weight}"
)

print(
    f"Threshold      : {best_threshold:.2f}"
)

print(
    f"Saved model    : {OUT_MODEL}"
)

print(
    f"Saved summary  : {OUT_SUMMARY}"
)

print(
    f"Saved split    : {OUT_SPLIT}"
)

# ============================================================
# Save WearVQA-only router
# ============================================================

WEAR_ONLY_MODEL = (
    ROOT
    / "routing/"
    / "e2b_wearvqa_failure_router.joblib"
)

wear_only_artifact = {
    "vectorizer": vectorizer,
    "model": model_wear,
    "threshold": float(
        result_wear["threshold"]
    ),
    "source": "WearVQA-only",
    "target_failure_recall": float(
        TARGET_FAILURE_RECALL
    ),
    "label_definition": {
        "CORRECT": 0,
        "PARTIAL": 0,
        "INCORRECT": 1,
        "AMBIGUOUS": "excluded",
    },
}

joblib.dump(
    wear_only_artifact,
    WEAR_ONLY_MODEL,
)

print()
print("=" * 90)
print("OVERALL BEST ROUTER")
print("=" * 90)
print("Model     : WearVQA-only")
print(
    f"Threshold : "
    f"{result_wear['threshold']:.2f}"
)
print(
    f"Saved     : "
    f"{WEAR_ONLY_MODEL}"
)
