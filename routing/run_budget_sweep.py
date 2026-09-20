from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


DATA = Path(
    "routing/benchmarks/router_training/"
    "paired_visual_router_dataset.csv"
)

OUT_DIR = Path(
    "routing/benchmarks/budget_sweep"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 40

BUDGETS = [
    0.25,
    0.50,
    0.75,
]

C = 1.0


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(DATA)

df["rescue_target"] = (
    (df["small_usable"] == 0)
    &
    (df["large_usable"] == 1)
).astype(int)


df["category"] = (
    df["category"]
    .fillna("unknown")
    .astype(str)
    .str.lower()
    .str.replace(
        r"[^a-z0-9]+",
        "_",
        regex=True,
    )
)

df["text_question_only"] = (
    df["question"].astype(str)
)

df["text_with_type"] = (
    "__CAT_"
    + df["category"]
    + "__ "
    + df["question"].astype(str)
)


train = df[
    df["split"] == "train"
].copy()

val = df[
    df["split"] == "val"
].copy()

test = df[
    df["split"] == "test"
].copy()


print("=" * 100)
print("DATA")
print("=" * 100)

print("Train:", len(train))
print("Val  :", len(val))
print("Test :", len(test))

print(
    "Test rescue rate:",
    f"{test['rescue_target'].mean()*100:.2f}%"
)


# ============================================================
# HELPERS
# ============================================================

def train_model(text_col):

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=70000,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    X_train = vectorizer.fit_transform(
        train[text_col].astype(str)
    )

    X_val = vectorizer.transform(
        val[text_col].astype(str)
    )

    X_test = vectorizer.transform(
        test[text_col].astype(str)
    )

    model = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=4000,
        solver="liblinear",
        random_state=SEED,
    )

    model.fit(
        X_train,
        train["rescue_target"],
    )

    return (
        model.predict_proba(X_val)[:, 1],
        model.predict_proba(X_test)[:, 1],
    )


def evaluate(
    choose_large,
    data,
):

    choose_large = np.asarray(
        choose_large,
        dtype=bool,
    )

    usable = np.where(
        choose_large,
        data["large_usable"],
        data["small_usable"],
    ).astype(float)

    latency = np.where(
        choose_large,
        data["large_latency"],
        data["small_latency"],
    ).astype(float)

    rescue_mask = (
        data["rescue_target"]
        == 1
    ).to_numpy()

    rescue_capture = (
        choose_large[
            rescue_mask
        ].mean()
        if rescue_mask.any()
        else np.nan
    )

    return {
        "usable_rate":
            usable.mean(),

        "large_rate":
            choose_large.mean(),

        "rescue_capture":
            rescue_capture,

        "mean_latency":
            latency.mean(),

        "median_latency":
            np.median(latency),

        "p95_latency":
            np.quantile(
                latency,
                0.95,
            ),
    }


def select_threshold(
    p_val,
    budget,
):
    """
    On validation:
    among thresholds satisfying cloud usage <= budget,
    choose the one with highest usable rate.

    Tie-break:
    lower LARGE usage.
    """

    candidates = []

    for threshold in np.arange(
        0.0,
        1.001,
        0.0025,
    ):

        choose_large = (
            p_val >= threshold
        )

        large_rate = (
            choose_large.mean()
        )

        if (
            large_rate
            > budget + 1e-9
        ):
            continue

        metrics = evaluate(
            choose_large,
            val,
        )

        candidates.append({
            "threshold":
                threshold,
            **metrics,
        })


    if not candidates:
        raise RuntimeError(
            f"No threshold for budget {budget}"
        )


    candidates = sorted(
        candidates,
        key=lambda x: (
            -x["usable_rate"],
            x["large_rate"],
        )
    )

    return candidates[0]


# ============================================================
# TRAIN BOTH VARIANTS
# ============================================================

experiments = {
    "Question only":
        "text_question_only",

    "Question + type":
        "text_with_type",
}


all_rows = []


for experiment_name, text_col in experiments.items():

    print()
    print("=" * 100)
    print(experiment_name.upper())
    print("=" * 100)

    p_val, p_test = train_model(
        text_col
    )


    for budget in BUDGETS:

        selected = select_threshold(
            p_val,
            budget,
        )

        threshold = selected[
            "threshold"
        ]

        choose_test = (
            p_test >= threshold
        )

        test_metrics = evaluate(
            choose_test,
            test,
        )


        row = {
            "model":
                experiment_name,

            "budget":
                budget,

            "threshold":
                threshold,

            "val_large_rate":
                selected[
                    "large_rate"
                ],

            "val_usable":
                selected[
                    "usable_rate"
                ],

            **{
                f"test_{k}": v
                for k, v
                in test_metrics.items()
            },
        }

        all_rows.append(row)


        print()
        print(
            f"Budget {budget*100:.0f}%"
        )

        print(
            f"  threshold      : "
            f"{threshold:.3f}"
        )

        print(
            f"  test LARGE     : "
            f"{test_metrics['large_rate']*100:.2f}%"
        )

        print(
            f"  test usable    : "
            f"{test_metrics['usable_rate']*100:.2f}%"
        )

        print(
            f"  rescue capture : "
            f"{test_metrics['rescue_capture']*100:.2f}%"
        )

        print(
            f"  mean latency   : "
            f"{test_metrics['mean_latency']:.3f}s"
        )


# ============================================================
# BASELINES
# ============================================================

always_small = evaluate(
    np.zeros(
        len(test),
        dtype=bool,
    ),
    test,
)

always_large = evaluate(
    np.ones(
        len(test),
        dtype=bool,
    ),
    test,
)

oracle_large = (
    test["rescue_target"]
    == 1
).to_numpy()

oracle = evaluate(
    oracle_large,
    test,
)


print()
print("=" * 100)
print("BASELINES")
print("=" * 100)

print(
    "Always SMALL:",
    f"usable={always_small['usable_rate']*100:.2f}%",
    f"LARGE={always_small['large_rate']*100:.2f}%",
    f"latency={always_small['mean_latency']:.3f}s",
)

print(
    "Always LARGE:",
    f"usable={always_large['usable_rate']*100:.2f}%",
    f"LARGE={always_large['large_rate']*100:.2f}%",
    f"latency={always_large['mean_latency']:.3f}s",
)

print(
    "Oracle rescue:",
    f"usable={oracle['usable_rate']*100:.2f}%",
    f"LARGE={oracle['large_rate']*100:.2f}%",
    f"latency={oracle['mean_latency']:.3f}s",
)


# ============================================================
# TABLE
# ============================================================

results = pd.DataFrame(
    all_rows
)

results.to_csv(
    OUT_DIR
    / "budget_sweep_results.csv",
    index=False,
)


print()
print("=" * 100)
print("FINAL TABLE")
print("=" * 100)

display = results[
    [
        "model",
        "budget",
        "test_large_rate",
        "test_usable_rate",
        "test_rescue_capture",
        "test_mean_latency",
    ]
].copy()

for c in [
    "budget",
    "test_large_rate",
    "test_usable_rate",
    "test_rescue_capture",
]:
    display[c] *= 100

print(
    display.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.2f}",
    )
)


# ============================================================
# PLOT 1:
# CLOUD USAGE vs USABLE RATE
# ============================================================

plt.figure(
    figsize=(7, 5)
)

for name in experiments:

    x = results[
        results["model"] == name
    ]

    plt.plot(
        x["test_large_rate"] * 100,
        x["test_usable_rate"] * 100,
        marker="o",
        label=name,
    )


plt.scatter(
    [0],
    [
        always_small[
            "usable_rate"
        ] * 100
    ],
    marker="x",
    s=100,
    label="Always SMALL",
)

plt.scatter(
    [100],
    [
        always_large[
            "usable_rate"
        ] * 100
    ],
    marker="x",
    s=100,
    label="Always LARGE",
)

plt.scatter(
    [
        oracle[
            "large_rate"
        ] * 100
    ],
    [
        oracle[
            "usable_rate"
        ] * 100
    ],
    marker="*",
    s=180,
    label="Oracle rescue",
)


plt.xlabel(
    "Cloud / LARGE usage (%)"
)

plt.ylabel(
    "Usable answer rate (%)"
)

plt.title(
    "Routing Trade-off: Cloud Usage vs Usable Accuracy"
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUT_DIR
    / "cloud_usage_vs_usable.png",
    dpi=200,
)


# ============================================================
# PLOT 2:
# CLOUD USAGE vs RESCUE CAPTURE
# ============================================================

plt.figure(
    figsize=(7, 5)
)

for name in experiments:

    x = results[
        results["model"] == name
    ]

    plt.plot(
        x["test_large_rate"] * 100,
        x["test_rescue_capture"] * 100,
        marker="o",
        label=name,
    )


plt.xlabel(
    "Cloud / LARGE usage (%)"
)

plt.ylabel(
    "LARGE rescue capture (%)"
)

plt.title(
    "Cloud Budget vs Rescue Capture"
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUT_DIR
    / "cloud_usage_vs_rescue.png",
    dpi=200,
)


print()
print("=" * 100)
print("SAVED")
print("=" * 100)

print(
    OUT_DIR
    / "budget_sweep_results.csv"
)

print(
    OUT_DIR
    / "cloud_usage_vs_usable.png"
)

print(
    OUT_DIR
    / "cloud_usage_vs_rescue.png"
)
