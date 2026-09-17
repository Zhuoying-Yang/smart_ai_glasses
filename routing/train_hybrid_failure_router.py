import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from datasets import load_dataset
from scipy.sparse import csr_matrix, hstack

from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    accuracy_score,
)


# ============================================================
# Split E2B benchmark by image
# ============================================================

def group_split(df, seed=40):
    groups = df["image_id"].astype(str).values

    split1 = GroupShuffleSplit(
        n_splits=1,
        train_size=0.70,
        random_state=seed,
    )

    train_idx, temp_idx = next(
        split1.split(df, groups=groups)
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    temp_groups = temp_df["image_id"].astype(str).values

    split2 = GroupShuffleSplit(
        n_splits=1,
        train_size=0.50,
        random_state=seed + 1,
    )

    val_idx, test_idx = next(
        split2.split(temp_df, groups=temp_groups)
    )

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df


# ============================================================
# Threshold selection
# ============================================================

def choose_threshold(y_true, prob, target_recall=0.85):
    candidates = []

    for threshold in np.linspace(0.01, 0.99, 197):
        pred = (prob >= threshold).astype(int)

        recall = recall_score(
            y_true,
            pred,
            zero_division=0,
        )

        precision = precision_score(
            y_true,
            pred,
            zero_division=0,
        )

        large_rate = pred.mean()

        candidates.append({
            "threshold": float(threshold),
            "recall": float(recall),
            "precision": float(precision),
            "large_rate": float(large_rate),
        })

    feasible = [
        x for x in candidates
        if x["recall"] >= target_recall
    ]

    if feasible:
        return min(
            feasible,
            key=lambda x: (
                x["large_rate"],
                -x["precision"],
            ),
        )

    return max(
        candidates,
        key=lambda x: (
            x["recall"],
            -x["large_rate"],
        ),
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate(name, y, prob, threshold):
    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    ).ravel()

    total = len(y)
    failures = int((y == 1).sum())
    successes = int((y == 0).sum())

    small_n = int((pred == 0).sum())
    large_n = int((pred == 1).sum())

    metrics = {
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
        "balanced_accuracy":
            balanced_accuracy_score(y, pred),
        "precision":
            precision_score(y, pred, zero_division=0),
        "recall":
            recall_score(y, pred, zero_division=0),
        "f1":
            f1_score(y, pred, zero_division=0),
        "large_rate": large_n / total,
        "over_escalation":
            fp / successes if successes else 0,
        "small_failure_rate":
            fn / small_n if small_n else 0,
    }

    print()
    print("=" * 78)
    print(name)
    print("=" * 78)

    print(f"ROC-AUC                  : {metrics['roc_auc']:.4f}")
    print(f"PR-AUC                   : {metrics['pr_auc']:.4f}")
    print(f"Balanced accuracy        : {metrics['balanced_accuracy']:.4f}")
    print(f"Failure precision        : {metrics['precision']:.4f}")
    print(f"Failure recall           : {metrics['recall']:.4f}")
    print(f"Failure F1               : {metrics['f1']:.4f}")

    print()
    print(f"Threshold                : {threshold:.3f}")
    print(
        f"Route SMALL              : "
        f"{small_n:,} ({small_n/total*100:.2f}%)"
    )
    print(
        f"Route LARGE              : "
        f"{large_n:,} ({large_n/total*100:.2f}%)"
    )

    print(
        f"Failures caught          : "
        f"{tp}/{failures} ({tp/failures*100:.2f}%)"
    )

    print(
        f"Over-escalation          : "
        f"{fp}/{successes} "
        f"({metrics['over_escalation']*100:.2f}%)"
    )

    print(
        f"Failure rate kept SMALL  : "
        f"{metrics['small_failure_rate']*100:.2f}%"
    )

    print()
    print("                         Pred SMALL   Pred LARGE")
    print(f"Actual success           {tn:>10}   {fp:>10}")
    print(f"Actual failure           {fn:>10}   {tp:>10}")

    return metrics


# ============================================================
# Text encoder
# ============================================================

def build_text_encoder():
    return FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=50000,
                sublinear_tf=True,
                strip_accents="unicode",
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                lowercase=True,
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=50000,
                sublinear_tf=True,
            ),
        ),
    ])


# ============================================================
# Train best binary LR using validation ROC-AUC
# ============================================================

def fit_best_lr(
    X_train,
    y_train,
    X_val,
    y_val,
    seed,
):
    best_model = None
    best_c = None
    best_auc = -1

    for C in [0.1, 0.3, 1.0, 3.0]:
        model = LogisticRegression(
            C=C,
            max_iter=2500,
            solver="liblinear",
            class_weight="balanced",
            random_state=seed,
        )

        model.fit(X_train, y_train)

        prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, prob)

        print(f"  C={C:<4} validation AUC={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_c = C

    print(
        f"  -> selected C={best_c}, "
        f"val AUC={best_auc:.4f}"
    )

    return best_model


# ============================================================
# Train a DISJOINT 65-class question classifier
#
# Important:
# exclude ALL images used in the 10k E2B benchmark.
# ============================================================

def train_disjoint_type_classifier(
    benchmark_df,
    n=30000,
    seed=40,
):
    print()
    print("=" * 78)
    print("TRAINING DISJOINT 65-CLASS QUESTION CLASSIFIER")
    print("=" * 78)

    print("Loading cached VQAv2 validation metadata...")

    ds = load_dataset(
        "parquet",
        data_files={
            "validation":
                "hf://datasets/lmms-lab/VQAv2/"
                "data/validation-*.parquet"
        },
        split="validation",
    )

    metadata = pd.DataFrame({
        "question": ds["question"],
        "question_type": ds["question_type"],
        "image_id": ds["image_id"],
    })

    benchmark_images = set(
        benchmark_df["image_id"].astype(int)
    )

    metadata = metadata[
        ~metadata["image_id"]
        .astype(int)
        .isin(benchmark_images)
    ].copy()

    print(
        f"Available disjoint questions : "
        f"{len(metadata):,}"
    )

    metadata = metadata.sample(
        n=min(n, len(metadata)),
        random_state=seed + 100,
    ).reset_index(drop=True)

    print(
        f"Selected type-classifier data: "
        f"{len(metadata):,}"
    )

    print(
        f"Classes                      : "
        f"{metadata['question_type'].nunique()}"
    )

    train_df, test_df = train_test_split(
        metadata,
        test_size=0.10,
        random_state=seed,
        stratify=metadata["question_type"],
    )

    vectorizer = build_text_encoder()

    print("Building type-classifier TF-IDF...")

    X_train = vectorizer.fit_transform(
        train_df["question"].astype(str)
    )

    X_test = vectorizer.transform(
        test_df["question"].astype(str)
    )

    clf = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        max_iter=2000,
        tol=1e-4,
        class_weight="balanced",
        random_state=seed,
    )

    print("Training question-type classifier...")

    clf.fit(
        X_train,
        train_df["question_type"],
    )

    pred = clf.predict(X_test)

    acc = accuracy_score(
        test_df["question_type"],
        pred,
    )

    macro_f1 = f1_score(
        test_df["question_type"],
        pred,
        average="macro",
    )

    print(
        f"Disjoint type classifier accuracy : "
        f"{acc:.4f}"
    )

    print(
        f"Disjoint type classifier Macro-F1 : "
        f"{macro_f1:.4f}"
    )

    return vectorizer, clf, acc, macro_f1


# ============================================================
# Load existing SigLIP embedding cache
# ============================================================

def load_image_cache(path):
    data = np.load(
        path,
        allow_pickle=False,
    )

    ids = data["image_ids"]
    emb = data["embeddings"]

    return {
        int(image_id): emb[i]
        for i, image_id in enumerate(ids)
    }


def image_matrix(df, embedding_map):
    return np.vstack([
        embedding_map[int(image_id)]
        for image_id in df["image_id"]
    ]).astype(np.float32)


# ============================================================
# Structured question-type features
# ============================================================

def type_probabilities(
    df,
    vectorizer,
    classifier,
):
    X = vectorizer.transform(
        df["question"].astype(str)
    )

    return classifier.predict_proba(X)


def compute_type_risks(
    train_probs,
    y_train,
):
    hard_type = np.argmax(
        train_probs,
        axis=1,
    )

    k = train_probs.shape[1]

    risks = np.zeros(k, dtype=np.float32)

    for i in range(k):
        mask = hard_type == i

        n = int(mask.sum())
        failures = int(
            y_train[mask].sum()
        )

        # Laplace smoothing
        risks[i] = (
            failures + 1
        ) / (
            n + 2
        )

    return risks


def build_type_features(
    probs,
    risk_vector,
):
    confidence = probs.max(
        axis=1,
        keepdims=True,
    )

    entropy = -np.sum(
        probs * np.log(probs + 1e-12),
        axis=1,
        keepdims=True,
    )

    entropy /= np.log(
        probs.shape[1]
    )

    expected_risk = (
        probs @ risk_vector
    ).reshape(-1, 1)

    return np.hstack([
        probs,
        confidence,
        entropy,
        expected_risk,
    ]).astype(np.float32)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "benchmark/results/final_10k/"
            "e2b_all_10015_cases.csv"
        ),
    )

    parser.add_argument(
        "--siglip-cache",
        default=(
            "benchmark/results/final_10k/"
            "siglip_image_embeddings.npz"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "routing/"
            "e2b_hybrid_failure_router.joblib"
        ),
    )

    parser.add_argument(
        "--target-failure-recall",
        type=float,
        default=0.85,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=40,
    )

    args = parser.parse_args()

    # ========================================================
    # Load E2B benchmark
    # ========================================================

    df = pd.read_csv(args.input)

    df = df.dropna(
        subset=[
            "question",
            "image_id",
            "vqa_score",
        ]
    ).copy()

    df["failure"] = (
        df["vqa_score"].astype(float)
        < 0.5
    ).astype(int)

    print("=" * 78)
    print("HYBRID E2B FAILURE ROUTER")
    print("=" * 78)

    print(f"Cases    : {len(df):,}")
    print(
        f"Failures : "
        f"{df['failure'].sum():,}"
    )

    train_df, val_df, test_df = group_split(
        df,
        seed=args.seed,
    )

    y_train = train_df["failure"].values
    y_val = val_df["failure"].values
    y_test = test_df["failure"].values

    print()
    print(
        f"Train / Val / Test : "
        f"{len(train_df):,} / "
        f"{len(val_df):,} / "
        f"{len(test_df):,}"
    )

    # ========================================================
    # Train disjoint 65-class classifier
    # ========================================================

    (
        type_vectorizer,
        type_classifier,
        type_acc,
        type_macro_f1,
    ) = train_disjoint_type_classifier(
        benchmark_df=df,
        n=30000,
        seed=args.seed,
    )

    # ========================================================
    # Type probabilities
    # ========================================================

    print()
    print("Generating 65-class type probabilities...")

    train_type_prob = type_probabilities(
        train_df,
        type_vectorizer,
        type_classifier,
    )

    val_type_prob = type_probabilities(
        val_df,
        type_vectorizer,
        type_classifier,
    )

    test_type_prob = type_probabilities(
        test_df,
        type_vectorizer,
        type_classifier,
    )

    # ========================================================
    # Per-type empirical failure risk
    # TRAIN ONLY
    # ========================================================

    type_risk_vector = compute_type_risks(
        train_type_prob,
        y_train,
    )

    train_type = build_type_features(
        train_type_prob,
        type_risk_vector,
    )

    val_type = build_type_features(
        val_type_prob,
        type_risk_vector,
    )

    test_type = build_type_features(
        test_type_prob,
        type_risk_vector,
    )

    # Expected risk is final column
    val_expected_risk = val_type[:, -1]
    test_expected_risk = test_type[:, -1]

    # ========================================================
    # TYPE-RISK baseline
    # ========================================================

    type_threshold_info = choose_threshold(
        y_val,
        val_expected_risk,
        args.target_failure_recall,
    )

    type_threshold = type_threshold_info[
        "threshold"
    ]

    type_metrics = evaluate(
        "TYPE-RISK ONLY HELD-OUT TEST",
        y_test,
        test_expected_risk,
        type_threshold,
    )

    # ========================================================
    # Text features
    # ========================================================

    print()
    print("Building failure-router text features...")

    text_encoder = build_text_encoder()

    X_train_text = text_encoder.fit_transform(
        train_df["question"].astype(str)
    )

    X_val_text = text_encoder.transform(
        val_df["question"].astype(str)
    )

    X_test_text = text_encoder.transform(
        test_df["question"].astype(str)
    )

    # ========================================================
    # Image features
    # ========================================================

    print("Loading cached SigLIP embeddings...")

    embedding_map = load_image_cache(
        args.siglip_cache
    )

    train_img = image_matrix(
        train_df,
        embedding_map,
    )

    val_img = image_matrix(
        val_df,
        embedding_map,
    )

    test_img = image_matrix(
        test_df,
        embedding_map,
    )

    image_scaler = StandardScaler()

    train_img = image_scaler.fit_transform(
        train_img
    )

    val_img = image_scaler.transform(
        val_img
    )

    test_img = image_scaler.transform(
        test_img
    )

    # ========================================================
    # Scale structured type features
    # ========================================================

    type_scaler = StandardScaler()

    train_type_scaled = (
        type_scaler.fit_transform(
            train_type
        )
    )

    val_type_scaled = (
        type_scaler.transform(
            val_type
        )
    )

    test_type_scaled = (
        type_scaler.transform(
            test_type
        )
    )

    # ========================================================
    # TEXT
    # ========================================================

    print()
    print("Training TEXT model...")

    text_model = fit_best_lr(
        X_train_text,
        y_train,
        X_val_text,
        y_val,
        args.seed,
    )

    val_prob = text_model.predict_proba(
        X_val_text
    )[:, 1]

    threshold = choose_threshold(
        y_val,
        val_prob,
        args.target_failure_recall,
    )["threshold"]

    test_prob_text = text_model.predict_proba(
        X_test_text
    )[:, 1]

    text_metrics = evaluate(
        "TEXT-ONLY HELD-OUT TEST",
        y_test,
        test_prob_text,
        threshold,
    )

    # ========================================================
    # TEXT + IMAGE
    # ========================================================

    X_train_ti = hstack([
        X_train_text,
        csr_matrix(train_img),
    ]).tocsr()

    X_val_ti = hstack([
        X_val_text,
        csr_matrix(val_img),
    ]).tocsr()

    X_test_ti = hstack([
        X_test_text,
        csr_matrix(test_img),
    ]).tocsr()

    print()
    print("Training TEXT + IMAGE model...")

    ti_model = fit_best_lr(
        X_train_ti,
        y_train,
        X_val_ti,
        y_val,
        args.seed,
    )

    val_prob = ti_model.predict_proba(
        X_val_ti
    )[:, 1]

    ti_threshold = choose_threshold(
        y_val,
        val_prob,
        args.target_failure_recall,
    )["threshold"]

    test_prob_ti = ti_model.predict_proba(
        X_test_ti
    )[:, 1]

    ti_metrics = evaluate(
        "TEXT + IMAGE HELD-OUT TEST",
        y_test,
        test_prob_ti,
        ti_threshold,
    )

    # ========================================================
    # FULL HYBRID
    # text + image + 65-type probabilities + risk
    # ========================================================

    X_train_hybrid = hstack([
        X_train_text,
        csr_matrix(train_img),
        csr_matrix(train_type_scaled),
    ]).tocsr()

    X_val_hybrid = hstack([
        X_val_text,
        csr_matrix(val_img),
        csr_matrix(val_type_scaled),
    ]).tocsr()

    X_test_hybrid = hstack([
        X_test_text,
        csr_matrix(test_img),
        csr_matrix(test_type_scaled),
    ]).tocsr()

    print()
    print("Training FULL HYBRID model...")

    hybrid_model = fit_best_lr(
        X_train_hybrid,
        y_train,
        X_val_hybrid,
        y_val,
        args.seed,
    )

    val_prob_hybrid = (
        hybrid_model.predict_proba(
            X_val_hybrid
        )[:, 1]
    )

    hybrid_threshold = choose_threshold(
        y_val,
        val_prob_hybrid,
        args.target_failure_recall,
    )["threshold"]

    test_prob_hybrid = (
        hybrid_model.predict_proba(
            X_test_hybrid
        )[:, 1]
    )

    hybrid_metrics = evaluate(
        "FULL HYBRID HELD-OUT TEST",
        y_test,
        test_prob_hybrid,
        hybrid_threshold,
    )

    # ========================================================
    # Comparison
    # ========================================================

    print()
    print("=" * 92)
    print("FINAL COMPARISON")
    print("=" * 92)

    print(
        f"{'Metric':<27}"
        f"{'Type Risk':>13}"
        f"{'Text':>13}"
        f"{'Text+Image':>15}"
        f"{'Hybrid':>13}"
    )

    print("-" * 92)

    metrics_to_show = [
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
        ("Failure recall", "recall"),
        ("LARGE rate", "large_rate"),
        ("Over-escalation", "over_escalation"),
        ("SMALL failure rate", "small_failure_rate"),
    ]

    for label, key in metrics_to_show:
        print(
            f"{label:<27}"
            f"{type_metrics[key]:>13.4f}"
            f"{text_metrics[key]:>13.4f}"
            f"{ti_metrics[key]:>15.4f}"
            f"{hybrid_metrics[key]:>13.4f}"
        )

    # ========================================================
    # Save final hybrid router
    # ========================================================

    bundle = {
        "text_encoder": text_encoder,

        "type_vectorizer":
            type_vectorizer,

        "type_classifier":
            type_classifier,

        "type_classes":
            type_classifier.classes_.tolist(),

        "type_risk_vector":
            type_risk_vector,

        "image_scaler":
            image_scaler,

        "type_scaler":
            type_scaler,

        "router":
            hybrid_model,

        "threshold":
            float(hybrid_threshold),

        "siglip_model":
            "google/siglip-base-patch16-224",

        "failure_definition":
            "vqa_score < 0.5",

        "target_failure_recall":
            args.target_failure_recall,

        "type_classifier_accuracy":
            float(type_acc),

        "type_classifier_macro_f1":
            float(type_macro_f1),

        "metrics": {
            "type_risk":
                type_metrics,
            "text":
                text_metrics,
            "text_image":
                ti_metrics,
            "hybrid":
                hybrid_metrics,
        },
    }

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        bundle,
        output_path,
    )

    # ========================================================
    # Save test predictions
    # ========================================================

    out = test_df.copy()

    out["type_risk"] = (
        test_expected_risk
    )

    out["text_failure_prob"] = (
        test_prob_text
    )

    out["text_image_failure_prob"] = (
        test_prob_ti
    )

    out["hybrid_failure_prob"] = (
        test_prob_hybrid
    )

    out["hybrid_route"] = np.where(
        test_prob_hybrid
        >= hybrid_threshold,
        "LARGE",
        "SMALL",
    )

    pred_path = Path(
        "benchmark/results/final_10k/"
        "hybrid_router_test_predictions.csv"
    )

    out.to_csv(
        pred_path,
        index=False,
    )

    print()
    print("=" * 92)
    print("SAVED")
    print("=" * 92)

    print(
        f"Hybrid router : "
        f"{output_path}"
    )

    print(
        f"Predictions   : "
        f"{pred_path}"
    )


if __name__ == "__main__":
    main()
