from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


DATA = Path(
    "routing/benchmarks/router_training/"
    "paired_visual_router_dataset.csv"
)

MODEL_OUT = Path(
    "routing/visual_rescue_router_budget50.joblib"
)

SEED = 40
MAX_LARGE_RATE = 0.50


df = pd.read_csv(DATA)

df["rescue_target"] = (
    (df["small_usable"] == 0)
    &
    (df["large_usable"] == 1)
).astype(int)


train = df[df["split"] == "train"].copy()
val = df[df["split"] == "val"].copy()
test = df[df["split"] == "test"].copy()


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


model = LogisticRegression(
    C=1.0,
    class_weight="balanced",
    max_iter=4000,
    solver="liblinear",
    random_state=SEED,
)

model.fit(
    X_train,
    train["rescue_target"],
)


p_val = model.predict_proba(X_val)[:, 1]


best = None

for threshold in np.arange(
    0.20,
    0.81,
    0.005,
):

    choose_large = (
        p_val >= threshold
    )

    large_rate = choose_large.mean()

    if large_rate > MAX_LARGE_RATE:
        continue

    routed_usable = np.where(
        choose_large,
        val["large_usable"],
        val["small_usable"],
    ).astype(float)

    usable_rate = routed_usable.mean()

    rescue_mask = (
        val["rescue_target"]
        == 1
    ).to_numpy()

    rescue_capture = (
        choose_large[rescue_mask].mean()
        if rescue_mask.any()
        else np.nan
    )

    candidate = {
        "threshold": threshold,
        "large_rate": large_rate,
        "usable_rate": usable_rate,
        "rescue_capture": rescue_capture,
    }

    if (
        best is None
        or usable_rate > best["usable_rate"]
    ):
        best = candidate


if best is None:
    raise RuntimeError(
        "No threshold satisfied LARGE-rate constraint."
    )


print("=" * 80)
print("VALIDATION")
print("=" * 80)

print(
    f"Threshold      : {best['threshold']:.3f}"
)

print(
    f"LARGE rate     : {best['large_rate']*100:.2f}%"
)

print(
    f"Usable rate    : {best['usable_rate']*100:.2f}%"
)

print(
    f"Rescue capture : {best['rescue_capture']*100:.2f}%"
)


# ============================================================
# TEST
# ============================================================

p_test = model.predict_proba(
    X_test
)[:, 1]

choose_large = (
    p_test
    >= best["threshold"]
)


routed_usable = np.where(
    choose_large,
    test["large_usable"],
    test["small_usable"],
).astype(float)

routed_latency = np.where(
    choose_large,
    test["large_latency"],
    test["small_latency"],
).astype(float)


rescue_mask = (
    test["rescue_target"]
    == 1
).to_numpy()

rescue_capture = (
    choose_large[
        rescue_mask
    ].mean()
)


print()
print("=" * 80)
print("TEST")
print("=" * 80)

print(
    f"Router usable  : {routed_usable.mean()*100:.2f}%"
)

print(
    f"Mean latency   : {routed_latency.mean():.3f}s"
)

print(
    f"Median latency : {np.median(routed_latency):.3f}s"
)

print(
    f"P95 latency    : {np.quantile(routed_latency, .95):.3f}s"
)

print(
    f"LARGE rate     : {choose_large.mean()*100:.2f}%"
)

print(
    f"Rescue capture : {rescue_capture*100:.2f}%"
)

print()
print(
    f"Always SMALL usable: "
    f"{test['small_usable'].mean()*100:.2f}%"
)

print(
    f"Always LARGE usable: "
    f"{test['large_usable'].mean()*100:.2f}%"
)


# ============================================================
# PER DATASET
# ============================================================

out = test.copy()

out["choose_large"] = (
    choose_large.astype(int)
)

out["routed_usable"] = (
    routed_usable
)

out["routed_latency"] = (
    routed_latency
)


print()
print("=" * 80)
print("PER DATASET")
print("=" * 80)

for dataset, x in out.groupby(
    "dataset"
):

    print()
    print(dataset)

    print(
        "Usable     :",
        f"{x['routed_usable'].mean()*100:.2f}%"
    )

    print(
        "LARGE rate :",
        f"{x['choose_large'].mean()*100:.2f}%"
    )

    print(
        "Latency    :",
        f"{x['routed_latency'].mean():.3f}s"
    )


# ============================================================
# REFIT TRAIN + VAL
# ============================================================

trainval = df[
    df["split"].isin(
        ["train", "val"]
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

X_tv = final_vectorizer.fit_transform(
    trainval["question"].astype(str)
)

final_model = LogisticRegression(
    C=1.0,
    class_weight="balanced",
    max_iter=4000,
    solver="liblinear",
    random_state=SEED,
)

final_model.fit(
    X_tv,
    trainval["rescue_target"],
)


artifact = {
    "version":
        "visual_rescue_router_budget50_v1",

    "vectorizer":
        final_vectorizer,

    "model":
        final_model,

    "threshold":
        float(best["threshold"]),

    "max_large_rate":
        MAX_LARGE_RATE,

    "positive_definition":
        "SMALL unusable AND LARGE usable",

    "action_positive":
        "LARGE_1F",

    "action_negative":
        "SMALL_1F",
}

joblib.dump(
    artifact,
    MODEL_OUT,
)

print()
print("Saved:", MODEL_OUT)
