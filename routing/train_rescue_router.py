from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# CONFIG
# ============================================================

INPUT = Path(
    "routing/benchmarks/router_training/"
    "paired_visual_router_dataset.csv"
)

OUT_DIR = Path(
    "routing/benchmarks/rescue_router"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_OUT = Path(
    "routing/visual_rescue_router_final.joblib"
)

SEED = 40

C_VALUES = [
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
]

THRESHOLDS = np.arange(
    0.20,
    0.81,
    0.02,
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT)

required = [
    "question",
    "split",
    "small_usable",
    "large_usable",
    "small_latency",
    "large_latency",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )


# ------------------------------------------------------------
# Rescue target
#
# 1 = SMALL fails AND LARGE works
# 0 = everything else
# ------------------------------------------------------------

df["rescue_target"] = (
    (df["small_usable"] == 0)
    &
    (df["large_usable"] == 1)
).astype(int)


train = df[
    df["split"] == "train"
].copy()

val = df[
    df["split"] == "val"
].copy()

test = df[
    df["split"] == "test"
].copy()


print("=" * 90)
print("RESCUE ROUTER DATA")
print("=" * 90)

print(
    "Train:",
    len(train),
)

print(
    "Val  :",
    len(val),
)

print(
    "Test :",
    len(test),
)

print()

print(
    "Overall rescue rate:",
    f"{df['rescue_target'].mean()*100:.2f}%"
)

print(
    "Train rescue rate  :",
    f"{train['rescue_target'].mean()*100:.2f}%"
)

print(
    "Val rescue rate    :",
    f"{val['rescue_target'].mean()*100:.2f}%"
)

print(
    "Test rescue rate   :",
    f"{test['rescue_target'].mean()*100:.2f}%"
)


# ============================================================
# FEATURES
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=70000,
    sublinear_tf=True,
    strip_accents="unicode",
)

X_train = vectorizer.fit_transform(
    train["question"].astype(str)
)

X_val = vectorizer.transform(
    val["question"].astype(str)
)

X_test = vectorizer.transform(
    test["question"].astype(str)
)

y_train = train[
    "rescue_target"
].to_numpy()

y_val = val[
    "rescue_target"
].to_numpy()

y_test = test[
    "rescue_target"
].to_numpy()


# ============================================================
# TUNE C + THRESHOLD ON VALIDATION
# ============================================================

best = None
rows = []

print()
print("=" * 90)
print("VALIDATION SEARCH")
print("=" * 90)

for C in C_VALUES:

    model = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=4000,
        solver="liblinear",
        random_state=SEED,
    )

    model.fit(
        X_train,
        y_train,
    )

    p_val = model.predict_proba(
        X_val
    )[:, 1]

    auc = roc_auc_score(
        y_val,
        p_val,
    )

    for threshold in THRESHOLDS:

        pred = (
            p_val >= threshold
        ).astype(int)

        precision = precision_score(
            y_val,
            pred,
            zero_division=0,
        )

        recall = recall_score(
            y_val,
            pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_val,
            pred,
            zero_division=0,
        )

        large_rate = pred.mean()

        row = {
            "C": C,
            "threshold": float(threshold),
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "large_rate": large_rate,
        }

        rows.append(row)

        if (
            best is None
            or f1 > best["f1"]
        ):
            best = {
                **row,
                "model": model,
            }


search_df = pd.DataFrame(rows)

search_df.to_csv(
    OUT_DIR
    / "validation_threshold_search.csv",
    index=False,
)


print(
    f"Best C          : {best['C']}"
)

print(
    f"Best threshold  : {best['threshold']:.2f}"
)

print(
    f"Validation AUC  : {best['auc']:.4f}"
)

print(
    f"Precision       : {best['precision']:.4f}"
)

print(
    f"Recall          : {best['recall']:.4f}"
)

print(
    f"F1              : {best['f1']:.4f}"
)

print(
    f"LARGE rate      : {best['large_rate']*100:.2f}%"
)


# ============================================================
# TEST
# ============================================================

model = best[
    "model"
]

threshold = best[
    "threshold"
]

p_test = model.predict_proba(
    X_test
)[:, 1]

choose_large = (
    p_test >= threshold
)

pred_test = choose_large.astype(int)


precision = precision_score(
    y_test,
    pred_test,
    zero_division=0,
)

recall = recall_score(
    y_test,
    pred_test,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    pred_test,
    zero_division=0,
)

auc = roc_auc_score(
    y_test,
    p_test,
)


# ------------------------------------------------------------
# Actual routed performance
# ------------------------------------------------------------

router_usable = np.where(
    choose_large,
    test["large_usable"],
    test["small_usable"],
).astype(float)

router_latency = np.where(
    choose_large,
    test["large_latency"],
    test["small_latency"],
).astype(float)


# Baselines
always_small_usable = (
    test["small_usable"]
    .to_numpy(dtype=float)
)

always_large_usable = (
    test["large_usable"]
    .to_numpy(dtype=float)
)

always_small_latency = (
    test["small_latency"]
    .to_numpy(dtype=float)
)

always_large_latency = (
    test["large_latency"]
    .to_numpy(dtype=float)
)


# Oracle rescue router:
# only escalate when LARGE genuinely rescues SMALL.
oracle_large = (
    (
        test["small_usable"] == 0
    )
    &
    (
        test["large_usable"] == 1
    )
).to_numpy()

oracle_usable = np.where(
    oracle_large,
    test["large_usable"],
    test["small_usable"],
).astype(float)

oracle_latency = np.where(
    oracle_large,
    test["large_latency"],
    test["small_latency"],
).astype(float)


# Rescue capture:
actual_rescue = (
    y_test == 1
)

rescue_capture = (
    choose_large[
        actual_rescue
    ].mean()
    if actual_rescue.any()
    else np.nan
)


# Unnecessary LARGE:
# LARGE chosen when it was NOT a rescue case.
unnecessary_large = (
    choose_large
    &
    (~actual_rescue)
)

unnecessary_large_rate = (
    unnecessary_large.mean()
)


def metrics(
    name,
    usable,
    latency,
    large_mask,
):
    return {
        "policy": name,
        "usable_rate":
            float(
                np.mean(usable)
            ),
        "mean_latency":
            float(
                np.mean(latency)
            ),
        "median_latency":
            float(
                np.median(latency)
            ),
        "p95_latency":
            float(
                np.quantile(
                    latency,
                    0.95,
                )
            ),
        "large_rate":
            float(
                np.mean(large_mask)
            ),
    }


results = []

results.append(
    metrics(
        "Always SMALL",
        always_small_usable,
        always_small_latency,
        np.zeros(
            len(test),
            dtype=bool,
        ),
    )
)

results.append(
    metrics(
        "Always LARGE",
        always_large_usable,
        always_large_latency,
        np.ones(
            len(test),
            dtype=bool,
        ),
    )
)

results.append(
    metrics(
        "Rescue router",
        router_usable,
        router_latency,
        choose_large,
    )
)

results.append(
    metrics(
        "Oracle rescue router",
        oracle_usable,
        oracle_latency,
        oracle_large,
    )
)


results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUT_DIR
    / "test_policy_comparison.csv",
    index=False,
)


print()
print("=" * 90)
print("RESCUE CLASSIFICATION — TEST")
print("=" * 90)

print(
    f"AUC             : {auc:.4f}"
)

print(
    f"Precision       : {precision:.4f}"
)

print(
    f"Recall          : {recall:.4f}"
)

print(
    f"F1              : {f1:.4f}"
)

print(
    f"Rescue capture  : {rescue_capture*100:.2f}%"
)

print(
    f"LARGE rate      : {choose_large.mean()*100:.2f}%"
)

print(
    f"Unnecessary LARGE rate: "
    f"{unnecessary_large_rate*100:.2f}%"
)


print()
print("=" * 90)
print("ROUTING POLICY — TEST")
print("=" * 90)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# ============================================================
# PER DATASET
# ============================================================

print()
print("=" * 90)
print("PER-DATASET ROUTER RESULTS")
print("=" * 90)

test_out = test.copy()

test_out[
    "rescue_probability"
] = p_test

test_out[
    "choose_large"
] = choose_large.astype(int)

test_out[
    "routed_usable"
] = router_usable

test_out[
    "routed_latency"
] = router_latency


for dataset, x in test_out.groupby(
    "dataset"
):

    print()
    print(dataset)

    print(
        "N             :",
        len(x),
    )

    print(
        "Usable        :",
        f"{x['routed_usable'].mean()*100:.2f}%"
    )

    print(
        "Mean latency  :",
        f"{x['routed_latency'].mean():.3f}s"
    )

    print(
        "LARGE rate    :",
        f"{x['choose_large'].mean()*100:.2f}%"
    )

    rescue_rows = (
        x["rescue_target"] == 1
    )

    if rescue_rows.any():

        print(
            "Rescue capture:",
            f"{x.loc[rescue_rows, 'choose_large'].mean()*100:.2f}%"
        )


test_out.to_csv(
    OUT_DIR
    / "test_predictions.csv",
    index=False,
)


# ============================================================
# REFIT FINAL MODEL ON TRAIN + VAL
# ============================================================

trainval = df[
    df["split"].isin(
        [
            "train",
            "val",
        ]
    )
].copy()


final_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=70000,
    sublinear_tf=True,
    strip_accents="unicode",
)

X_trainval = (
    final_vectorizer
    .fit_transform(
        trainval[
            "question"
        ].astype(str)
    )
)

y_trainval = (
    trainval[
        "rescue_target"
    ].to_numpy()
)


final_model = LogisticRegression(
    C=best["C"],
    class_weight="balanced",
    max_iter=4000,
    solver="liblinear",
    random_state=SEED,
)

final_model.fit(
    X_trainval,
    y_trainval,
)


artifact = {
    "version":
        "visual_rescue_router_v1",

    "vectorizer":
        final_vectorizer,

    "model":
        final_model,

    "threshold":
        float(
            threshold
        ),

    "positive_definition":
        "SMALL unusable AND LARGE usable",

    "action_positive":
        "LARGE_1F",

    "action_negative":
        "SMALL_1F",

    "C":
        float(
            best["C"]
        ),

    "training_rows":
        int(
            len(trainval)
        ),
}


joblib.dump(
    artifact,
    MODEL_OUT,
)


print()
print("=" * 90)
print("DONE")
print("=" * 90)

print(
    "Final model:",
    MODEL_OUT,
)

print(
    "Policy results:",
    OUT_DIR
    / "test_policy_comparison.csv",
)

print(
    "Test predictions:",
    OUT_DIR
    / "test_predictions.csv",
)

print(
    "Threshold search:",
    OUT_DIR
    / "validation_threshold_search.csv",
)
