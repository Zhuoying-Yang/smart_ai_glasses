from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(".")
OUT = Path("routing/benchmarks/router_training")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 40

# Equal importance as discussed:
ACCURACY_WEIGHT = 0.50
LATENCY_WEIGHT = 0.50

LABEL_TO_USABLE = {
    "CORRECT": 1.0,
    "PARTIAL": 1.0,
    "INCORRECT": 0.0,
    "AMBIGUOUS": np.nan,
}

C_VALUES = [0.5, 1.0, 2.0, 4.0, 8.0]
RIDGE_ALPHAS = [1.0, 10.0, 100.0]

# LARGE latency for SuperGlasses uses our final deployment config:
# Luna-medium + resize1024.
#
# WearVQA LARGE inference was run with original image detail, so for the
# latency model we prefer SuperGlasses because it matches the final config.
LARGE_LATENCY_SUPERGLASSES_ONLY = True


# ============================================================
# FILE LOCATIONS
# ============================================================

FILES = {
    "wearvqa": {
        "small_labels": [
            Path(
                "routing/benchmarks/final_sol_labels/"
                "wearvqa_e2b_sol_graded.csv"
            ),
            Path(
                "routing/benchmarks/results/"
                "wearvqa_e2b_full_graded.csv"
            ),
        ],
        "small_latency": [
            Path(
                "routing/benchmarks/results/"
                "wearvqa_e2b_full_raw.csv"
            ),
        ],
        "large_labels": [
            Path(
                "routing/benchmarks/results/"
                "wearvqa_luna_sol_image_aware_graded.csv"
            ),
        ],
        "large_latency": [
            Path(
                "routing/benchmarks/results/"
                "wearvqa_luna_full_raw.csv"
            ),
        ],
    },

    "superglasses": {
        "small_labels": [
            Path(
                "routing/benchmarks/final_sol_labels/"
                "superglasses_e2b_sol_graded.csv"
            ),
            Path(
                "routing/benchmarks/superglasses/results/"
                "superglasses_direct_visual_e2b_graded.csv"
            ),
        ],
        "small_latency": [
            Path(
                "routing/benchmarks/superglasses/results/"
                "superglasses_direct_visual_e2b.csv"
            ),
        ],
        "large_labels": [
            Path(
                "routing/benchmarks/superglasses/results/"
                "superglasses_large_luna_medium_1024_sol_graded.csv"
            ),
        ],
        "large_latency": [
            Path(
                "routing/benchmarks/superglasses/results/"
                "superglasses_large_luna_medium_1024.csv"
            ),
        ],
    },
}


# ============================================================
# HELPERS
# ============================================================

def first_existing(paths, required=True):
    for p in paths:
        if p.exists():
            return p

    if required:
        raise FileNotFoundError(
            "Could not find any of:\n"
            + "\n".join(str(x) for x in paths)
        )

    return None


def read_csv(path):
    df = pd.read_csv(path)

    if "sample_id" not in df.columns:
        raise ValueError(
            f"{path} has no sample_id column"
        )

    df["sample_id"] = (
        df["sample_id"]
        .astype(str)
        .str.strip()
    )

    return df


def first_column(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def normalize_label(x):
    if pd.isna(x):
        return np.nan

    x = str(x).strip().upper()

    if x in LABEL_TO_USABLE:
        return x

    return np.nan


def normalize_question(q):
    q = str(q).lower()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[^\w\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def mode_value(series):
    m = series.mode()
    if len(m):
        return m.iloc[0]
    return series.iloc[0]


def extract_latency(df, side):
    if side == "small":
        candidates = [
            "e2b_latency_s",
            "small_latency_s",
            "latency_sec",
            "latency_s",
            "latency",
        ]
    else:
        candidates = [
            "luna_latency_s",
            "large_latency_s",
            "latency_sec",
            "latency_s",
            "latency",
        ]

    col = first_column(
        df,
        candidates,
    )

    if col is None:
        return pd.Series(
            np.nan,
            index=df.index,
        )

    return pd.to_numeric(
        df[col],
        errors="coerce",
    )


def load_side(
    label_path,
    latency_path,
    side,
):
    df = read_csv(
        label_path
    )

    question_col = first_column(
        df,
        [
            "question",
            "query",
        ],
    )

    if question_col is None:
        raise ValueError(
            f"No question column in {label_path}"
        )

    category_col = first_column(
        df,
        [
            "category",
            "question_type",
            "task",
            "domain",
        ],
    )

    # Prefer human label when present.
    human_col = first_column(
        df,
        [
            "human_label",
            "review_label",
        ],
    )

    judge_col = first_column(
        df,
        [
            "judge_label",
            "final_label",
            "label",
            "sol_label",
            "final_sol_label",
            "grade",
        ],
    )

    # Fallback: automatically find a column whose non-empty
    # values are grading labels.
    if judge_col is None:
        valid_grade_values = {
            "CORRECT",
            "PARTIAL",
            "INCORRECT",
            "AMBIGUOUS",
        }

        for col in df.columns:
            values = set(
                df[col]
                .dropna()
                .astype(str)
                .str.strip()
                .str.upper()
                .unique()
            )

            if (
                values
                and values.issubset(valid_grade_values)
            ):
                judge_col = col
                break

    if judge_col is None and human_col is None:
        print("\nColumns available in:", label_path)
        print(list(df.columns))
        raise ValueError(
            f"No grading label column in {label_path}"
        )

    print(
        f"  Detected {side.upper()} label column:",
        judge_col if judge_col is not None else human_col,
    )

    if human_col is not None:
        human = (
            df[human_col]
            .map(normalize_label)
            .astype("object")
        )
    else:
        human = pd.Series(
            [None] * len(df),
            index=df.index,
            dtype="object",
        )

    if judge_col is not None:
        judge = (
            df[judge_col]
            .map(normalize_label)
            .astype("object")
        )
    else:
        judge = pd.Series(
            [None] * len(df),
            index=df.index,
            dtype="object",
        )

    # Prefer a human label when available;
    # otherwise use the automatic judge label.
    final_label = (
        human
        .combine_first(judge)
        .astype("object")
    )

    # If a sample is explicitly pending human review,
    # do not silently treat it as training truth unless
    # a human label has actually been entered.
    if (
        "review_status"
        in df.columns
    ):
        needs_review = (
            df["review_status"]
            .fillna("")
            .astype(str)
            .str.contains(
                "NEEDS_HUMAN_REVIEW",
                case=False,
            )
        )

        no_human = human.isna()

        final_label.loc[
            needs_review & no_human
        ] = np.nan

    latency = extract_latency(
        df,
        side,
    )

    out = pd.DataFrame(
        {
            "sample_id":
                df["sample_id"],

            "question":
                df[question_col]
                .astype(str),

            "category":
                (
                    df[category_col]
                    .astype(str)
                    if category_col
                    else "unknown"
                ),

            f"{side}_label":
                final_label,

            f"{side}_latency":
                latency,
        }
    )

    out[
        f"{side}_usable"
    ] = out[
        f"{side}_label"
    ].map(
        LABEL_TO_USABLE
    )

    # If grading CSV did not retain inference latency,
    # merge from the raw inference result.
    if (
        out[
            f"{side}_latency"
        ].notna().sum()
        == 0
        and latency_path is not None
        and latency_path.exists()
    ):
        raw = read_csv(
            latency_path
        )

        raw_latency = extract_latency(
            raw,
            side,
        )

        lat = pd.DataFrame(
            {
                "sample_id":
                    raw["sample_id"],

                "_latency":
                    raw_latency,
            }
        )

        lat = (
            lat
            .dropna(
                subset=[
                    "_latency"
                ]
            )
            .drop_duplicates(
                "sample_id",
                keep="last",
            )
        )

        out = out.merge(
            lat,
            on="sample_id",
            how="left",
        )

        out[
            f"{side}_latency"
        ] = out[
            f"{side}_latency"
        ].fillna(
            out["_latency"]
        )

        out = out.drop(
            columns=[
                "_latency"
            ]
        )

    return out


def make_features():
    word = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=60000,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    char = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=40000,
        sublinear_tf=True,
    )

    return FeatureUnion(
        [
            ("word", word),
            ("char", char),
        ]
    )


def fit_best_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    name,
):
    best = None

    for C in C_VALUES:
        model = LogisticRegression(
            C=C,
            max_iter=4000,
            solver="liblinear",
            random_state=SEED,
        )

        model.fit(
            X_train,
            y_train,
        )

        p = model.predict_proba(
            X_val
        )[:, 1]

        brier = brier_score_loss(
            y_val,
            p,
        )

        try:
            auc = roc_auc_score(
                y_val,
                p,
            )
        except Exception:
            auc = float("nan")

        print(
            f"{name}: "
            f"C={C:<4} "
            f"Brier={brier:.4f} "
            f"AUC={auc:.4f}"
        )

        candidate = {
            "C": C,
            "brier": brier,
            "auc": auc,
            "model": model,
        }

        if (
            best is None
            or candidate[
                "brier"
            ] < best[
                "brier"
            ]
        ):
            best = candidate

    print(
        f"Selected {name} C="
        f"{best['C']}"
    )

    return best


def fit_best_latency(
    X_train,
    y_train,
    X_val,
    y_val,
    name,
):
    y_train_log = np.log1p(
        y_train
    )

    best = None

    for alpha in RIDGE_ALPHAS:
        model = Ridge(
            alpha=alpha,
            solver="lsqr",
        )

        model.fit(
            X_train,
            y_train_log,
        )

        pred = np.maximum(
            0.05,
            np.expm1(
                model.predict(
                    X_val
                )
            ),
        )

        mae = mean_absolute_error(
            y_val,
            pred,
        )

        print(
            f"{name}: "
            f"alpha={alpha:<6} "
            f"MAE={mae:.3f}s"
        )

        candidate = {
            "alpha": alpha,
            "mae": mae,
            "model": model,
        }

        if (
            best is None
            or mae < best["mae"]
        ):
            best = candidate

    print(
        f"Selected {name} alpha="
        f"{best['alpha']}"
    )

    return best


def latency_utility(
    latency,
    scale,
):
    return (
        1.0
        - np.clip(
            np.asarray(
                latency,
                dtype=float,
            )
            / scale,
            0.0,
            1.0,
        )
    )


def actual_utility(
    usable,
    latency,
    scale,
):
    return (
        ACCURACY_WEIGHT
        * np.asarray(
            usable,
            dtype=float,
        )
        +
        LATENCY_WEIGHT
        * latency_utility(
            latency,
            scale,
        )
    )


def evaluate_policy(
    df,
    choose_large,
    latency_scale,
    name,
):
    choose_large = np.asarray(
        choose_large,
        dtype=bool,
    )

    chosen_usable = np.where(
        choose_large,
        df["large_usable"],
        df["small_usable"],
    ).astype(float)

    chosen_latency = np.where(
        choose_large,
        df["large_latency"],
        df["small_latency"],
    ).astype(float)

    utility = actual_utility(
        chosen_usable,
        chosen_latency,
        latency_scale,
    )

    rescue_mask = (
        (df["small_usable"] == 0)
        &
        (df["large_usable"] == 1)
    ).to_numpy()

    both_mask = (
        (df["small_usable"] == 1)
        &
        (df["large_usable"] == 1)
    ).to_numpy()

    rescue_capture = (
        choose_large[
            rescue_mask
        ].mean()
        if rescue_mask.any()
        else np.nan
    )

    both_large_rate = (
        choose_large[
            both_mask
        ].mean()
        if both_mask.any()
        else np.nan
    )

    return {
        "policy":
            name,

        "n":
            len(df),

        "usable_rate":
            chosen_usable.mean(),

        "mean_latency":
            chosen_latency.mean(),

        "median_latency":
            np.median(
                chosen_latency
            ),

        "p95_latency":
            np.quantile(
                chosen_latency,
                0.95,
            ),

        "large_rate":
            choose_large.mean(),

        "mean_equal_utility":
            utility.mean(),

        "large_rescue_capture":
            rescue_capture,

        "large_rate_when_both_work":
            both_large_rate,
    }


# ============================================================
# BUILD PAIRED DATASET
# ============================================================

all_pairs = []

print()
print("=" * 100)
print("LOADING DATA")
print("=" * 100)

for dataset, cfg in FILES.items():

    small_label_path = first_existing(
        cfg["small_labels"]
    )

    large_label_path = first_existing(
        cfg["large_labels"]
    )

    small_latency_path = first_existing(
        cfg["small_latency"],
        required=False,
    )

    large_latency_path = first_existing(
        cfg["large_latency"],
        required=False,
    )

    print()
    print(dataset)
    print("  SMALL labels :", small_label_path)
    print("  SMALL latency:", small_latency_path)
    print("  LARGE labels :", large_label_path)
    print("  LARGE latency:", large_latency_path)

    small = load_side(
        small_label_path,
        small_latency_path,
        "small",
    )

    large = load_side(
        large_label_path,
        large_latency_path,
        "large",
    )

    pair = small.merge(
        large,
        on="sample_id",
        how="inner",
        suffixes=(
            "_small",
            "_large",
        ),
    )

    pair["dataset"] = dataset

    pair["question"] = (
        pair["question_small"]
        .fillna(
            pair[
                "question_large"
            ]
        )
    )

    pair["category"] = (
        pair["category_small"]
        .fillna(
            pair[
                "category_large"
            ]
        )
    )

    all_pairs.append(
        pair[
            [
                "dataset",
                "sample_id",
                "question",
                "category",
                "small_label",
                "small_usable",
                "small_latency",
                "large_label",
                "large_usable",
                "large_latency",
            ]
        ]
    )


paired_all = pd.concat(
    all_pairs,
    ignore_index=True,
)

paired_all[
    "unique_id"
] = (
    paired_all["dataset"]
    + ":"
    + paired_all["sample_id"]
)

paired_all.to_csv(
    OUT
    / "paired_visual_all.csv",
    index=False,
)


# Training/evaluation requires:
#   - both grading outcomes known
#   - both latencies known
paired = paired_all[
    paired_all[
        "small_usable"
    ].notna()
    &
    paired_all[
        "large_usable"
    ].notna()
    &
    paired_all[
        "small_latency"
    ].notna()
    &
    paired_all[
        "large_latency"
    ].notna()
    &
    (
        paired_all[
            "small_latency"
        ] > 0
    )
    &
    (
        paired_all[
            "large_latency"
        ] > 0
    )
].copy()


paired[
    "small_usable"
] = paired[
    "small_usable"
].astype(int)

paired[
    "large_usable"
] = paired[
    "large_usable"
].astype(int)


def pair_state(row):
    s = int(
        row["small_usable"]
    )

    l = int(
        row["large_usable"]
    )

    if s == 1 and l == 1:
        return "BOTH_WORK"

    if s == 1 and l == 0:
        return "SMALL_ONLY"

    if s == 0 and l == 1:
        return "LARGE_RESCUE"

    return "NEITHER"


paired[
    "pair_state"
] = paired.apply(
    pair_state,
    axis=1,
)

paired[
    "normalized_question"
] = paired[
    "question"
].map(
    normalize_question
)


print()
print("=" * 100)
print("PAIRED DATASET")
print("=" * 100)

print(
    "All matched rows       :",
    len(paired_all),
)

print(
    "Judgeable + latency    :",
    len(paired),
)

print()

print(
    paired[
        "dataset"
    ]
    .value_counts()
    .to_string()
)

print()
print("Pair states:")

print(
    paired[
        "pair_state"
    ]
    .value_counts()
    .to_string()
)


matrix = pd.crosstab(
    paired[
        "small_usable"
    ],
    paired[
        "large_usable"
    ],
    rownames=[
        "SMALL usable"
    ],
    colnames=[
        "LARGE usable"
    ],
)

print()
print("RESCUE MATRIX:")
print(matrix.to_string())


# Save category-level rescue analysis
category_stats = (
    paired
    .groupby(
        [
            "dataset",
            "category",
        ]
    )
    .agg(
        n=(
            "unique_id",
            "count",
        ),
        small_usable_rate=(
            "small_usable",
            "mean",
        ),
        large_usable_rate=(
            "large_usable",
            "mean",
        ),
        small_mean_latency=(
            "small_latency",
            "mean",
        ),
        large_mean_latency=(
            "large_latency",
            "mean",
        ),
    )
    .reset_index()
)

rescue = (
    paired[
        paired[
            "pair_state"
        ]
        == "LARGE_RESCUE"
    ]
    .groupby(
        [
            "dataset",
            "category",
        ]
    )
    .size()
    .rename(
        "large_rescue_count"
    )
    .reset_index()
)

category_stats = (
    category_stats
    .merge(
        rescue,
        on=[
            "dataset",
            "category",
        ],
        how="left",
    )
)

category_stats[
    "large_rescue_count"
] = category_stats[
    "large_rescue_count"
].fillna(0)

category_stats.to_csv(
    OUT
    / "rescue_by_category.csv",
    index=False,
)


# ============================================================
# GROUP-SAFE TRAIN / VAL / TEST SPLIT
# ============================================================

group_table = (
    paired
    .groupby(
        "normalized_question"
    )
    .agg(
        state=(
            "pair_state",
            mode_value,
        ),
        count=(
            "unique_id",
            "count",
        ),
    )
    .reset_index()
)


def safe_stratify(df):
    counts = (
        df["state"]
        .value_counts()
    )

    if len(counts) < 2:
        return None

    if counts.min() < 2:
        return None

    return df["state"]


train_groups, temp_groups = (
    train_test_split(
        group_table,
        test_size=0.20,
        random_state=SEED,
        stratify=safe_stratify(
            group_table
        ),
    )
)

val_groups, test_groups = (
    train_test_split(
        temp_groups,
        test_size=0.50,
        random_state=SEED,
        stratify=safe_stratify(
            temp_groups
        ),
    )
)


train_set = set(
    train_groups[
        "normalized_question"
    ]
)

val_set = set(
    val_groups[
        "normalized_question"
    ]
)

test_set = set(
    test_groups[
        "normalized_question"
    ]
)


def split_name(q):
    if q in train_set:
        return "train"

    if q in val_set:
        return "val"

    return "test"


paired[
    "split"
] = paired[
    "normalized_question"
].map(
    split_name
)

paired.to_csv(
    OUT
    / "paired_visual_router_dataset.csv",
    index=False,
)


train = paired[
    paired["split"]
    == "train"
].copy()

val = paired[
    paired["split"]
    == "val"
].copy()

test = paired[
    paired["split"]
    == "test"
].copy()


print()
print("=" * 100)
print("SPLIT")
print("=" * 100)

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


# ============================================================
# LATENCY NORMALIZATION
# ============================================================

train_latencies = np.concatenate(
    [
        train[
            "small_latency"
        ].to_numpy(
            dtype=float
        ),

        train[
            "large_latency"
        ].to_numpy(
            dtype=float
        ),
    ]
)

LATENCY_SCALE = float(
    np.quantile(
        train_latencies,
        0.95,
    )
)

print()
print(
    "Latency normalization scale "
    f"(train pooled P95): "
    f"{LATENCY_SCALE:.3f}s"
)


# ============================================================
# TEXT FEATURES
# ============================================================

features = make_features()

X_train = features.fit_transform(
    train["question"]
)

X_val = features.transform(
    val["question"]
)

X_test = features.transform(
    test["question"]
)


# ============================================================
# ACCURACY MODELS
# ============================================================

print()
print("=" * 100)
print("SMALL USABILITY MODEL")
print("=" * 100)

small_clf_info = fit_best_classifier(
    X_train,
    train[
        "small_usable"
    ].to_numpy(),
    X_val,
    val[
        "small_usable"
    ].to_numpy(),
    "SMALL",
)


print()
print("=" * 100)
print("LARGE USABILITY MODEL")
print("=" * 100)

large_clf_info = fit_best_classifier(
    X_train,
    train[
        "large_usable"
    ].to_numpy(),
    X_val,
    val[
        "large_usable"
    ].to_numpy(),
    "LARGE",
)


# ============================================================
# LATENCY MODELS
# ============================================================

print()
print("=" * 100)
print("SMALL LATENCY MODEL")
print("=" * 100)

small_lat_info = fit_best_latency(
    X_train,
    train[
        "small_latency"
    ].to_numpy(
        dtype=float
    ),
    X_val,
    val[
        "small_latency"
    ].to_numpy(
        dtype=float
    ),
    "SMALL latency",
)


print()
print("=" * 100)
print("LARGE LATENCY MODEL")
print("=" * 100)


if LARGE_LATENCY_SUPERGLASSES_ONLY:

    large_train_mask = (
        train["dataset"]
        == "superglasses"
    )

    large_val_mask = (
        val["dataset"]
        == "superglasses"
    )

    if (
        large_train_mask.sum()
        >= 100
        and large_val_mask.sum()
        >= 20
    ):
        X_large_train = (
            X_train[
                large_train_mask.to_numpy()
            ]
        )

        y_large_train = (
            train.loc[
                large_train_mask,
                "large_latency",
            ]
            .to_numpy(
                dtype=float
            )
        )

        X_large_val = (
            X_val[
                large_val_mask.to_numpy()
            ]
        )

        y_large_val = (
            val.loc[
                large_val_mask,
                "large_latency",
            ]
            .to_numpy(
                dtype=float
            )
        )

        print(
            "Using SuperGlasses-only "
            "LARGE latency because it "
            "matches Luna-medium + 1024."
        )

    else:
        X_large_train = X_train
        y_large_train = train[
            "large_latency"
        ].to_numpy(
            dtype=float
        )

        X_large_val = X_val
        y_large_val = val[
            "large_latency"
        ].to_numpy(
            dtype=float
        )

else:
    X_large_train = X_train
    y_large_train = train[
        "large_latency"
    ].to_numpy(
        dtype=float
    )

    X_large_val = X_val
    y_large_val = val[
        "large_latency"
    ].to_numpy(
        dtype=float
    )


large_lat_info = fit_best_latency(
    X_large_train,
    y_large_train,
    X_large_val,
    y_large_val,
    "LARGE latency",
)


# ============================================================
# DIRECT 50/50 UTILITY ROUTER
# ============================================================

train_small_u = actual_utility(
    train[
        "small_usable"
    ],
    train[
        "small_latency"
    ],
    LATENCY_SCALE,
)

train_large_u = actual_utility(
    train[
        "large_usable"
    ],
    train[
        "large_latency"
    ],
    LATENCY_SCALE,
)

y_direct_train = (
    train_large_u
    > train_small_u
).astype(int)


val_small_u = actual_utility(
    val[
        "small_usable"
    ],
    val[
        "small_latency"
    ],
    LATENCY_SCALE,
)

val_large_u = actual_utility(
    val[
        "large_usable"
    ],
    val[
        "large_latency"
    ],
    LATENCY_SCALE,
)


print()
print("=" * 100)
print("DIRECT EQUAL-WEIGHT ROUTER")
print("=" * 100)


best_direct = None

for C in C_VALUES:

    model = LogisticRegression(
        C=C,
        max_iter=4000,
        solver="liblinear",
        random_state=SEED,
    )

    model.fit(
        X_train,
        y_direct_train,
    )

    p = model.predict_proba(
        X_val
    )[:, 1]

    for threshold in np.arange(
        0.20,
        0.81,
        0.02,
    ):

        choose_large = (
            p >= threshold
        )

        realized = np.where(
            choose_large,
            val_large_u,
            val_small_u,
        ).mean()

        candidate = {
            "C": C,
            "threshold":
                float(threshold),
            "utility":
                float(realized),
            "large_rate":
                float(
                    choose_large.mean()
                ),
            "model": model,
        }

        if (
            best_direct is None
            or candidate[
                "utility"
            ]
            > best_direct[
                "utility"
            ]
        ):
            best_direct = candidate


print(
    "Selected direct router:",
    f"C={best_direct['C']}, "
    f"threshold="
    f"{best_direct['threshold']:.2f}, "
    f"val utility="
    f"{best_direct['utility']:.4f}, "
    f"val LARGE rate="
    f"{best_direct['large_rate']:.2%}",
)


# ============================================================
# TEST EVALUATION
# ============================================================

small_clf = small_clf_info[
    "model"
]

large_clf = large_clf_info[
    "model"
]

small_lat_model = small_lat_info[
    "model"
]

large_lat_model = large_lat_info[
    "model"
]

direct_model = best_direct[
    "model"
]


p_small = small_clf.predict_proba(
    X_test
)[:, 1]

p_large = large_clf.predict_proba(
    X_test
)[:, 1]


pred_small_latency = np.maximum(
    0.05,
    np.expm1(
        small_lat_model.predict(
            X_test
        )
    ),
)

pred_large_latency = np.maximum(
    0.05,
    np.expm1(
        large_lat_model.predict(
            X_test
        )
    ),
)


small_score = (
    ACCURACY_WEIGHT
    * p_small
    +
    LATENCY_WEIGHT
    * latency_utility(
        pred_small_latency,
        LATENCY_SCALE,
    )
)

large_score = (
    ACCURACY_WEIGHT
    * p_large
    +
    LATENCY_WEIGHT
    * latency_utility(
        pred_large_latency,
        LATENCY_SCALE,
    )
)


choose_decomposed = (
    large_score
    > small_score
)

choose_accuracy_only = (
    p_large
    > p_small
)

direct_probability = (
    direct_model
    .predict_proba(
        X_test
    )[:, 1]
)

choose_direct = (
    direct_probability
    >= best_direct[
        "threshold"
    ]
)


actual_small_test_u = actual_utility(
    test[
        "small_usable"
    ],
    test[
        "small_latency"
    ],
    LATENCY_SCALE,
)

actual_large_test_u = actual_utility(
    test[
        "large_usable"
    ],
    test[
        "large_latency"
    ],
    LATENCY_SCALE,
)

choose_oracle = (
    actual_large_test_u
    > actual_small_test_u
)


results = []

results.append(
    evaluate_policy(
        test,
        np.zeros(
            len(test),
            dtype=bool,
        ),
        LATENCY_SCALE,
        "Always SMALL",
    )
)

results.append(
    evaluate_policy(
        test,
        np.ones(
            len(test),
            dtype=bool,
        ),
        LATENCY_SCALE,
        "Always LARGE",
    )
)

results.append(
    evaluate_policy(
        test,
        choose_accuracy_only,
        LATENCY_SCALE,
        "Predicted accuracy only",
    )
)

results.append(
    evaluate_policy(
        test,
        choose_decomposed,
        LATENCY_SCALE,
        "Predicted accuracy + latency",
    )
)

results.append(
    evaluate_policy(
        test,
        choose_direct,
        LATENCY_SCALE,
        "Direct 50/50 utility router",
    )
)

results.append(
    evaluate_policy(
        test,
        choose_oracle,
        LATENCY_SCALE,
        "Oracle upper bound",
    )
)


results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUT
    / "test_policy_comparison.csv",
    index=False,
)


print()
print("=" * 100)
print("TEST POLICY COMPARISON")
print("=" * 100)

display_cols = [
    "policy",
    "usable_rate",
    "mean_latency",
    "median_latency",
    "p95_latency",
    "large_rate",
    "mean_equal_utility",
    "large_rescue_capture",
]

print(
    results_df[
        display_cols
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# Choose between the two latency-aware learned routers
# using VALIDATION, not test.
#
# Direct router already has best_direct utility.
# Evaluate decomposed router on validation too.

val_p_small = small_clf.predict_proba(
    X_val
)[:, 1]

val_p_large = large_clf.predict_proba(
    X_val
)[:, 1]

val_pred_small_lat = np.maximum(
    0.05,
    np.expm1(
        small_lat_model.predict(
            X_val
        )
    ),
)

val_pred_large_lat = np.maximum(
    0.05,
    np.expm1(
        large_lat_model.predict(
            X_val
        )
    ),
)

val_small_score = (
    ACCURACY_WEIGHT
    * val_p_small
    +
    LATENCY_WEIGHT
    * latency_utility(
        val_pred_small_lat,
        LATENCY_SCALE,
    )
)

val_large_score = (
    ACCURACY_WEIGHT
    * val_p_large
    +
    LATENCY_WEIGHT
    * latency_utility(
        val_pred_large_lat,
        LATENCY_SCALE,
    )
)

val_choose_decomp = (
    val_large_score
    > val_small_score
)

val_decomp_result = evaluate_policy(
    val,
    val_choose_decomp,
    LATENCY_SCALE,
    "decomposed",
)

if (
    best_direct["utility"]
    >= val_decomp_result[
        "mean_equal_utility"
    ]
):
    selected_method = (
        "direct_equal_utility"
    )
else:
    selected_method = (
        "decomposed_equal_utility"
    )


print()
print(
    "Selected final routing method:",
    selected_method,
)


# ============================================================
# SAVE TEST PREDICTIONS / ERRORS
# ============================================================

test_out = test.copy()

test_out[
    "pred_small_usable_prob"
] = p_small

test_out[
    "pred_large_usable_prob"
] = p_large

test_out[
    "pred_small_latency"
] = pred_small_latency

test_out[
    "pred_large_latency"
] = pred_large_latency

test_out[
    "pred_small_score"
] = small_score

test_out[
    "pred_large_score"
] = large_score

test_out[
    "direct_large_probability"
] = direct_probability

test_out[
    "choose_large_decomposed"
] = choose_decomposed.astype(int)

test_out[
    "choose_large_direct"
] = choose_direct.astype(int)

test_out.to_csv(
    OUT
    / "test_predictions.csv",
    index=False,
)


# ============================================================
# REFIT FINAL MODELS ON TRAIN + VAL
# ============================================================

trainval = paired[
    paired["split"].isin(
        [
            "train",
            "val",
        ]
    )
].copy()

final_features = make_features()

X_trainval = (
    final_features
    .fit_transform(
        trainval[
            "question"
        ]
    )
)


final_small_clf = (
    LogisticRegression(
        C=small_clf_info[
            "C"
        ],
        max_iter=4000,
        solver="liblinear",
        random_state=SEED,
    )
)

final_small_clf.fit(
    X_trainval,
    trainval[
        "small_usable"
    ],
)


final_large_clf = (
    LogisticRegression(
        C=large_clf_info[
            "C"
        ],
        max_iter=4000,
        solver="liblinear",
        random_state=SEED,
    )
)

final_large_clf.fit(
    X_trainval,
    trainval[
        "large_usable"
    ],
)


final_small_latency = Ridge(
    alpha=small_lat_info[
        "alpha"
    ],
    solver="lsqr",
)

final_small_latency.fit(
    X_trainval,
    np.log1p(
        trainval[
            "small_latency"
        ]
    ),
)


if LARGE_LATENCY_SUPERGLASSES_ONLY:
    large_tv_mask = (
        trainval[
            "dataset"
        ]
        == "superglasses"
    )

    if large_tv_mask.sum() >= 100:
        final_large_latency = Ridge(
            alpha=large_lat_info[
                "alpha"
            ],
            solver="lsqr",
        )

        final_large_latency.fit(
            X_trainval[
                large_tv_mask.to_numpy()
            ],
            np.log1p(
                trainval.loc[
                    large_tv_mask,
                    "large_latency",
                ]
            ),
        )
    else:
        final_large_latency = Ridge(
            alpha=large_lat_info[
                "alpha"
            ],
            solver="lsqr",
        )

        final_large_latency.fit(
            X_trainval,
            np.log1p(
                trainval[
                    "large_latency"
                ]
            ),
        )

else:
    final_large_latency = Ridge(
        alpha=large_lat_info[
            "alpha"
        ],
        solver="lsqr",
    )

    final_large_latency.fit(
        X_trainval,
        np.log1p(
            trainval[
                "large_latency"
            ]
        ),
    )


tv_small_u = actual_utility(
    trainval[
        "small_usable"
    ],
    trainval[
        "small_latency"
    ],
    LATENCY_SCALE,
)

tv_large_u = actual_utility(
    trainval[
        "large_usable"
    ],
    trainval[
        "large_latency"
    ],
    LATENCY_SCALE,
)

tv_direct_target = (
    tv_large_u
    > tv_small_u
).astype(int)


final_direct_model = (
    LogisticRegression(
        C=best_direct[
            "C"
        ],
        max_iter=4000,
        solver="liblinear",
        random_state=SEED,
    )
)

final_direct_model.fit(
    X_trainval,
    tv_direct_target,
)


artifact = {
    "version":
        "visual_small_large_router_v1",

    "features":
        final_features,

    "small_usable_model":
        final_small_clf,

    "large_usable_model":
        final_large_clf,

    "small_latency_model":
        final_small_latency,

    "large_latency_model":
        final_large_latency,

    "direct_utility_model":
        final_direct_model,

    "direct_threshold":
        float(
            best_direct[
                "threshold"
            ]
        ),

    "selected_method":
        selected_method,

    "accuracy_weight":
        ACCURACY_WEIGHT,

    "latency_weight":
        LATENCY_WEIGHT,

    "latency_scale_s":
        LATENCY_SCALE,

    "small_classifier_C":
        small_clf_info[
            "C"
        ],

    "large_classifier_C":
        large_clf_info[
            "C"
        ],

    "small_latency_alpha":
        small_lat_info[
            "alpha"
        ],

    "large_latency_alpha":
        large_lat_info[
            "alpha"
        ],

    "direct_C":
        best_direct[
            "C"
        ],

    "training_rows":
        len(trainval),

    "datasets":
        sorted(
            trainval[
                "dataset"
            ].unique()
        ),
}


MODEL_OUT = Path(
    "routing/"
    "visual_small_large_router_final.joblib"
)

joblib.dump(
    artifact,
    MODEL_OUT,
)


# ============================================================
# SAVE SUMMARY JSON
# ============================================================

summary = {
    "all_matched_rows":
        int(
            len(
                paired_all
            )
        ),

    "judgeable_paired_rows":
        int(
            len(
                paired
            )
        ),

    "train_rows":
        int(
            len(train)
        ),

    "val_rows":
        int(
            len(val)
        ),

    "test_rows":
        int(
            len(test)
        ),

    "pair_state_counts":
        {
            str(k): int(v)
            for k, v
            in paired[
                "pair_state"
            ]
            .value_counts()
            .items()
        },

    "latency_scale_s":
        LATENCY_SCALE,

    "accuracy_weight":
        ACCURACY_WEIGHT,

    "latency_weight":
        LATENCY_WEIGHT,

    "selected_method":
        selected_method,

    "test_results":
        results,
}


with (
    OUT
    / "training_summary.json"
).open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print()
print("=" * 100)
print("DONE")
print("=" * 100)

print(
    "Final router:",
    MODEL_OUT,
)

print(
    "Paired data :",
    OUT
    / "paired_visual_router_dataset.csv",
)

print(
    "Test metrics:",
    OUT
    / "test_policy_comparison.csv",
)

print(
    "Predictions :",
    OUT
    / "test_predictions.csv",
)

print(
    "Category analysis:",
    OUT
    / "rescue_by_category.csv",
)

print(
    "Summary:",
    OUT
    / "training_summary.json",
)
