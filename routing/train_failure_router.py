import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# Split by IMAGE, not by individual question.
#
# This prevents multiple questions from the same VQAv2 image
# from appearing in both train and test.
# ============================================================

def group_split(df, seed=40):
    groups = df["image_id"].astype(str).values

    # 70% train, 30% temporary
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

    # Split remaining 30% equally:
    # 15% validation, 15% test
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
#
# Goal:
# catch at least target_recall fraction of true E2B failures,
# while sending as few requests to LARGE as possible.
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
        # Minimum cloud usage while satisfying desired failure recall.
        best = min(
            feasible,
            key=lambda x: (
                x["large_rate"],
                -x["precision"],
            ),
        )
    else:
        # Fallback: maximize recall, then minimize cloud usage.
        best = max(
            candidates,
            key=lambda x: (
                x["recall"],
                -x["large_rate"],
            ),
        )

    return best


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

    success_count = int((y == 0).sum())
    failure_count = int((y == 1).sum())

    large_count = int(pred.sum())
    small_count = int((pred == 0).sum())

    failure_recall = (
        tp / failure_count
        if failure_count > 0
        else 0.0
    )

    escalation_precision = (
        tp / large_count
        if large_count > 0
        else 0.0
    )

    over_escalation_rate = (
        fp / success_count
        if success_count > 0
        else 0.0
    )

    missed_failure_rate = (
        fn / failure_count
        if failure_count > 0
        else 0.0
    )

    small_failure_rate = (
        fn / small_count
        if small_count > 0
        else 0.0
    )

    print()
    print("=" * 78)
    print(name)
    print("=" * 78)

    print(f"Cases                    : {total:,}")
    print(f"Actual E2B successes     : {success_count:,}")
    print(f"Actual E2B failures      : {failure_count:,}")
    print()

    print(f"ROC-AUC                  : {roc_auc_score(y, prob):.4f}")
    print(f"PR-AUC                   : {average_precision_score(y, prob):.4f}")
    print(f"Accuracy                 : {accuracy_score(y, pred):.4f}")
    print(f"Balanced accuracy        : {balanced_accuracy_score(y, pred):.4f}")
    print(f"Failure precision        : {precision_score(y, pred, zero_division=0):.4f}")
    print(f"Failure recall           : {recall_score(y, pred, zero_division=0):.4f}")
    print(f"Failure F1               : {f1_score(y, pred, zero_division=0):.4f}")
    print()

    print(f"Routing threshold        : {threshold:.3f}")
    print(f"Route SMALL              : {small_count:,} ({small_count/total*100:.2f}%)")
    print(f"Route LARGE              : {large_count:,} ({large_count/total*100:.2f}%)")
    print()

    print(f"Failures caught          : {tp}/{failure_count} ({failure_recall*100:.2f}%)")
    print(f"Failures missed          : {fn}/{failure_count} ({missed_failure_rate*100:.2f}%)")
    print(
        f"Successful SMALL cases unnecessarily escalated : "
        f"{fp}/{success_count} ({over_escalation_rate*100:.2f}%)"
    )
    print(
        f"Among LARGE routes, actual failures            : "
        f"{tp}/{large_count} ({escalation_precision*100:.2f}%)"
        if large_count
        else "Among LARGE routes, actual failures            : N/A"
    )
    print(
        f"Failure rate among requests kept on SMALL      : "
        f"{small_failure_rate*100:.2f}%"
    )

    print()
    print("Confusion matrix")
    print("                         Pred SMALL   Pred LARGE")
    print(f"Actual E2B success       {tn:>10}   {fp:>10}")
    print(f"Actual E2B failure       {fn:>10}   {tp:>10}")

    return {
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, pred)
        ),
        "failure_precision": float(
            precision_score(y, pred, zero_division=0)
        ),
        "failure_recall": float(
            recall_score(y, pred, zero_division=0)
        ),
        "failure_f1": float(
            f1_score(y, pred, zero_division=0)
        ),
        "large_rate": float(large_count / total),
        "small_rate": float(small_count / total),
        "small_failure_rate": float(small_failure_rate),
        "over_escalation_rate": float(over_escalation_rate),
    }


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
        "--output",
        default="routing/e2b_failure_router.joblib",
    )

    parser.add_argument(
        "--predictions",
        default=(
            "benchmark/results/final_10k/"
            "failure_router_test_predictions.csv"
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

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    required = {
        "question",
        "image_id",
        "vqa_score",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # Clean data
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "question",
            "image_id",
            "vqa_score",
        ]
    ).copy()

    # Binary target:
    #
    # 0 = E2B succeeds sufficiently
    # 1 = E2B failure
    #
    # VQA score < 0.5 is treated as failure.
    df["failure"] = (
        df["vqa_score"].astype(float) < 0.5
    ).astype(int)

    print("=" * 78)
    print("E2B FAILURE ROUTER TRAINING")
    print("=" * 78)

    print(f"Total usable cases : {len(df):,}")
    print(
        f"E2B success        : "
        f"{(df['failure'] == 0).sum():,}"
    )
    print(
        f"E2B failure        : "
        f"{(df['failure'] == 1).sum():,}"
    )
    print(
        f"Overall failure rate: "
        f"{df['failure'].mean()*100:.2f}%"
    )

    # --------------------------------------------------------
    # Train / validation / test
    # --------------------------------------------------------

    train_df, val_df, test_df = group_split(
        df,
        seed=args.seed,
    )

    print()
    print("Data split by image_id")
    print("-" * 78)
    print(f"Train : {len(train_df):,}")
    print(f"Val   : {len(val_df):,}")
    print(f"Test  : {len(test_df):,}")

    print()
    print(
        f"Train failure rate : "
        f"{train_df['failure'].mean()*100:.2f}%"
    )
    print(
        f"Val failure rate   : "
        f"{val_df['failure'].mean()*100:.2f}%"
    )
    print(
        f"Test failure rate  : "
        f"{test_df['failure'].mean()*100:.2f}%"
    )

    # --------------------------------------------------------
    # Lightweight question representation
    # --------------------------------------------------------

    features = FeatureUnion([
        (
            "word_tfidf",
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
            "char_tfidf",
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

    model = Pipeline([
        ("features", features),
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                max_iter=2000,
                solver="liblinear",
                class_weight="balanced",
                random_state=args.seed,
            ),
        ),
    ])

    print()
    print("Training TF-IDF + Logistic Regression router...")

    model.fit(
        train_df["question"].astype(str),
        train_df["failure"].values,
    )

    print("Training complete.")

    # --------------------------------------------------------
    # Validation threshold
    # --------------------------------------------------------

    val_prob = model.predict_proba(
        val_df["question"].astype(str)
    )[:, 1]

    threshold_info = choose_threshold(
        val_df["failure"].values,
        val_prob,
        target_recall=args.target_failure_recall,
    )

    threshold = threshold_info["threshold"]

    print()
    print("=" * 78)
    print("VALIDATION THRESHOLD SELECTION")
    print("=" * 78)

    print(
        f"Target failure recall : "
        f"{args.target_failure_recall:.2f}"
    )
    print(
        f"Selected threshold    : "
        f"{threshold:.3f}"
    )
    print(
        f"Validation recall     : "
        f"{threshold_info['recall']:.4f}"
    )
    print(
        f"Validation precision  : "
        f"{threshold_info['precision']:.4f}"
    )
    print(
        f"Validation LARGE rate : "
        f"{threshold_info['large_rate']*100:.2f}%"
    )

    evaluate(
        "VALIDATION RESULTS",
        val_df["failure"].values,
        val_prob,
        threshold,
    )

    # --------------------------------------------------------
    # Final held-out test
    # --------------------------------------------------------

    test_prob = model.predict_proba(
        test_df["question"].astype(str)
    )[:, 1]

    test_metrics = evaluate(
        "FINAL HELD-OUT TEST RESULTS",
        test_df["failure"].values,
        test_prob,
        threshold,
    )

    # --------------------------------------------------------
    # Save test predictions
    # --------------------------------------------------------

    pred_df = test_df.copy()

    pred_df["predicted_failure_probability"] = (
        test_prob
    )

    pred_df["route"] = np.where(
        test_prob >= threshold,
        "LARGE",
        "SMALL",
    )

    predictions_path = Path(args.predictions)
    predictions_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pred_df.to_csv(
        predictions_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save trained router
    # --------------------------------------------------------

    bundle = {
        "model": model,
        "threshold": float(threshold),
        "failure_definition": "vqa_score < 0.5",
        "target_failure_recall": float(
            args.target_failure_recall
        ),
        "seed": int(args.seed),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "test_metrics": test_metrics,
        "description": (
            "Question-only E2B failure router. "
            "Word+bigram TF-IDF plus char TF-IDF, "
            "Logistic Regression."
        ),
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

    print()
    print("=" * 78)
    print("SAVED")
    print("=" * 78)
    print(f"Router model     : {output_path}")
    print(f"Test predictions : {predictions_path}")

    # --------------------------------------------------------
    # Quick examples
    # --------------------------------------------------------

    examples = [
        "What is in front of me?",
        "How many objects are on the table?",
        "Why is the person doing that?",
        "What color is the cup?",
        "Where is the phone?",
        "Is there a laptop in front of me?",
    ]

    example_prob = model.predict_proba(
        examples
    )[:, 1]

    print()
    print("EXAMPLE ROUTING")
    print("-" * 78)

    for q, p in zip(examples, example_prob):
        route = (
            "LARGE"
            if p >= threshold
            else "SMALL"
        )

        print(
            f"{route:<5}  "
            f"risk={p:.3f}  "
            f"{q}"
        )


if __name__ == "__main__":
    main()
