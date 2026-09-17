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
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


INPUT = Path(
    "benchmark/results/final_10k/"
    "e2b_all_10015_cases.csv"
)

OUTPUT = Path(
    "routing/e2b_calibrated_failure_router.joblib"
)

SEED = 40
TARGET_RECALL = 0.85


# ============================================================
# Same image-level split as previous experiments
# ============================================================

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

    train = df.iloc[train_idx].reset_index(drop=True)
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

    val = temp.iloc[val_idx].reset_index(drop=True)
    test = temp.iloc[test_idx].reset_index(drop=True)

    return train, val, test


# ============================================================
# Threshold selection
# ============================================================

def choose_threshold(y, prob):
    feasible = []

    for threshold in np.linspace(0.01, 0.99, 197):
        pred = (prob >= threshold).astype(int)

        recall = recall_score(
            y,
            pred,
            zero_division=0,
        )

        precision = precision_score(
            y,
            pred,
            zero_division=0,
        )

        large_rate = pred.mean()

        if recall >= TARGET_RECALL:
            feasible.append(
                (
                    large_rate,
                    -precision,
                    threshold,
                    recall,
                    precision,
                )
            )

    if not feasible:
        return 0.5

    feasible.sort()

    return float(feasible[0][2])


def evaluate(name, y, prob, threshold):
    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    ).ravel()

    print()
    print("=" * 72)
    print(name)
    print("=" * 72)

    print(
        f"Mean predicted risk : "
        f"{prob.mean():.3f}"
    )

    print(
        f"Actual failure rate : "
        f"{y.mean():.3f}"
    )

    print(
        f"ROC-AUC             : "
        f"{roc_auc_score(y, prob):.4f}"
    )

    print(
        f"PR-AUC              : "
        f"{average_precision_score(y, prob):.4f}"
    )

    print(
        f"Brier score         : "
        f"{brier_score_loss(y, prob):.4f}"
    )

    print(
        f"Threshold           : "
        f"{threshold:.3f}"
    )

    print(
        f"Failure recall      : "
        f"{recall_score(y, pred):.4f}"
    )

    print(
        f"Failure precision   : "
        f"{precision_score(y, pred):.4f}"
    )

    print(
        f"Failure F1          : "
        f"{f1_score(y, pred):.4f}"
    )

    print(
        f"Route LARGE         : "
        f"{pred.mean()*100:.2f}%"
    )

    print(
        f"Failures caught     : "
        f"{tp}/{tp+fn}"
    )

    print(
        f"Success escalated   : "
        f"{fp}/{fp+tn}"
    )


# ============================================================
# Load benchmark
# ============================================================

df = pd.read_csv(INPUT)

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

train, val, test = group_split(df)

print("=" * 72)
print("CALIBRATED E2B FAILURE ROUTER")
print("=" * 72)

print(f"Train : {len(train):,}")
print(f"Val   : {len(val):,}")
print(f"Test  : {len(test):,}")

print(
    f"Train failure rate: "
    f"{train.failure.mean():.3f}"
)


# ============================================================
# Text representation
# ============================================================

vectorizer = FeatureUnion([
    (
        "word",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=3,

            # slightly stronger regularization against
            # extremely rare phrases
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

            # slightly less aggressive than previous version
            ngram_range=(3, 4),

            min_df=4,
            max_features=30000,
            sublinear_tf=True,
        ),
    ),
])

print()
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
# IMPORTANT:
# no class_weight="balanced"
#
# Then calibrate probability with 5-fold sigmoid calibration.
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

print("Training calibrated classifier...")

model.fit(
    X_train,
    y_train,
)

print("Training complete.")


# ============================================================
# Validation threshold
# ============================================================

val_prob = model.predict_proba(
    X_val
)[:, 1]

threshold = choose_threshold(
    y_val,
    val_prob,
)

evaluate(
    "VALIDATION",
    y_val,
    val_prob,
    threshold,
)


# ============================================================
# Final held-out test
# ============================================================

test_prob = model.predict_proba(
    X_test
)[:, 1]

evaluate(
    "HELD-OUT TEST",
    y_test,
    test_prob,
    threshold,
)


# ============================================================
# Example wearable questions
# ============================================================

questions = [
    "What is in front of me?",
    "How many objects are on the table?",
    "Why is the person doing that?",
    "What color is the cup?",
    "Where is the phone?",
    "Is there a laptop in front of me?",
]

X_example = vectorizer.transform(
    questions
)

example_prob = model.predict_proba(
    X_example
)[:, 1]

print()
print("=" * 72)
print("LIVE-STYLE EXAMPLES")
print("=" * 72)

for q, p in zip(
    questions,
    example_prob,
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
}

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

joblib.dump(
    bundle,
    OUTPUT,
)

print()
print("=" * 72)
print("SAVED")
print("=" * 72)
print(OUTPUT)
