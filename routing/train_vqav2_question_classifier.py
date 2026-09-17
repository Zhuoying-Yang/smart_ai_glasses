import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from datasets import load_dataset

from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=40)

    parser.add_argument(
        "--output",
        default="routing/vqav2_65class_question_classifier.joblib",
    )

    args = parser.parse_args()

    print("=" * 78)
    print("VQAv2 65-CLASS QUESTION TYPE CLASSIFIER")
    print("=" * 78)

    # ========================================================
    # Load only validation parquet
    # ========================================================

    print("Loading VQAv2 validation...")

    ds = load_dataset(
        "parquet",
        data_files={
            "validation":
                "hf://datasets/lmms-lab/VQAv2/"
                "data/validation-*.parquet"
        },
        split="validation",
    )

    print(f"Full validation size : {len(ds):,}")

    # Same reproducible random sampling
    ds = ds.shuffle(seed=args.seed).select(
        range(min(args.n, len(ds)))
    )

    df = pd.DataFrame({
        "question_id": ds["question_id"],
        "question": ds["question"],
        "question_type": ds["question_type"],
    })

    df = df.dropna().reset_index(drop=True)

    print(f"Selected questions   : {len(df):,}")
    print(
        f"Question types       : "
        f"{df['question_type'].nunique()}"
    )

    # ========================================================
    # Distribution
    # ========================================================

    counts = (
        df["question_type"]
        .value_counts()
        .sort_values(ascending=False)
    )

    print()
    print("Largest classes")
    print("-" * 78)
    print(counts.head(10).to_string())

    print()
    print("Smallest classes")
    print("-" * 78)
    print(counts.tail(10).to_string())

    # ========================================================
    # 80 / 10 / 10 split
    # ========================================================

    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=args.seed,
        stratify=df["question_type"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=args.seed + 1,
        stratify=temp_df["question_type"],
    )

    print()
    print("Split")
    print("-" * 78)
    print(f"Train : {len(train_df):,}")
    print(f"Val   : {len(val_df):,}")
    print(f"Test  : {len(test_df):,}")

    # ========================================================
    # TF-IDF
    #
    # Word ngrams:
    #   "how many"
    #   "what color"
    #   "where is"
    #
    # Char ngrams:
    #   gives some robustness to ASR / spelling variation
    # ========================================================

    vectorizer = FeatureUnion([
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

    print()
    print("Building TF-IDF features...")

    X_train = vectorizer.fit_transform(
        train_df["question"].astype(str)
    )

    X_val = vectorizer.transform(
        val_df["question"].astype(str)
    )

    X_test = vectorizer.transform(
        test_df["question"].astype(str)
    )

    print(f"Feature dimension : {X_train.shape[1]:,}")

    # ========================================================
    # Fast multiclass classifier
    # ========================================================

    clf = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        max_iter=2000,
        tol=1e-4,
        class_weight="balanced",
        random_state=args.seed,
    )

    print()
    print("Training 65-class classifier...")

    clf.fit(
        X_train,
        train_df["question_type"],
    )

    print("Training complete.")

    # ========================================================
    # Validation
    # ========================================================

    val_pred = clf.predict(X_val)

    print()
    print("=" * 78)
    print("VALIDATION")
    print("=" * 78)

    print(
        f"Accuracy          : "
        f"{accuracy_score(val_df['question_type'], val_pred):.4f}"
    )

    print(
        f"Balanced accuracy : "
        f"{balanced_accuracy_score(val_df['question_type'], val_pred):.4f}"
    )

    print(
        f"Macro-F1          : "
        f"{f1_score(val_df['question_type'], val_pred, average='macro'):.4f}"
    )

    # ========================================================
    # Held-out test
    # ========================================================

    test_pred = clf.predict(X_test)
    test_prob = clf.predict_proba(X_test)

    test_conf = test_prob.max(axis=1)

    test_acc = accuracy_score(
        test_df["question_type"],
        test_pred,
    )

    test_bal_acc = balanced_accuracy_score(
        test_df["question_type"],
        test_pred,
    )

    test_macro_f1 = f1_score(
        test_df["question_type"],
        test_pred,
        average="macro",
    )

    print()
    print("=" * 78)
    print("FINAL HELD-OUT TEST")
    print("=" * 78)

    print(f"Cases             : {len(test_df):,}")
    print(f"Classes           : {len(clf.classes_)}")
    print(f"Accuracy          : {test_acc:.4f}")
    print(f"Balanced accuracy : {test_bal_acc:.4f}")
    print(f"Macro-F1          : {test_macro_f1:.4f}")
    print(
        f"Mean confidence   : "
        f"{test_conf.mean():.4f}"
    )

    # ========================================================
    # Per-class report
    # ========================================================

    report = classification_report(
        test_df["question_type"],
        test_pred,
        output_dict=True,
        zero_division=0,
    )

    rows = []

    for cls in clf.classes_:
        r = report.get(cls, {})

        rows.append({
            "question_type": cls,
            "precision": r.get("precision", 0),
            "recall": r.get("recall", 0),
            "f1": r.get("f1-score", 0),
            "support": int(r.get("support", 0)),
        })

    per_class = pd.DataFrame(rows)

    per_class = per_class.sort_values(
        "f1",
        ascending=False,
    )

    # ========================================================
    # Predictions
    # ========================================================

    pred_df = test_df.copy()

    pred_df["predicted_type"] = test_pred
    pred_df["confidence"] = test_conf
    pred_df["correct"] = (
        pred_df["predicted_type"]
        == pred_df["question_type"]
    )

    out_dir = Path(
        "benchmark/results/question_classifier"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pred_path = (
        out_dir /
        "vqav2_65class_test_predictions.csv"
    )

    class_path = (
        out_dir /
        "vqav2_65class_per_class.csv"
    )

    dist_path = (
        out_dir /
        "vqav2_65class_distribution.csv"
    )

    pred_df.to_csv(
        pred_path,
        index=False,
    )

    per_class.to_csv(
        class_path,
        index=False,
    )

    counts.rename("n").to_csv(
        dist_path,
    )

    # ========================================================
    # Save model
    # ========================================================

    bundle = {
        "vectorizer": vectorizer,
        "classifier": clf,
        "classes": clf.classes_.tolist(),
        "seed": args.seed,
        "n_samples": len(df),
        "test_accuracy": float(test_acc),
        "test_balanced_accuracy": float(test_bal_acc),
        "test_macro_f1": float(test_macro_f1),
        "description": (
            "VQAv2 65-class question-form classifier. "
            "Word+bigram TF-IDF + char TF-IDF + "
            "multiclass logistic SGD."
        ),
    }

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        bundle,
        output,
    )

    # ========================================================
    # Best / worst classes
    # ========================================================

    print()
    print("BEST CLASSES BY F1")
    print("-" * 78)

    print(
        per_class.head(15)[
            [
                "question_type",
                "support",
                "precision",
                "recall",
                "f1",
            ]
        ].to_string(index=False)
    )

    print()
    print("WORST CLASSES BY F1")
    print("-" * 78)

    print(
        per_class.tail(15)
        .sort_values("f1")[
            [
                "question_type",
                "support",
                "precision",
                "recall",
                "f1",
            ]
        ].to_string(index=False)
    )

    # ========================================================
    # Live-style examples
    # ========================================================

    examples = [
        "What is in front of me?",
        "How many objects are on the table?",
        "Why is the person doing that?",
        "What color is the cup?",
        "Where is the phone?",
        "Is there a laptop in front of me?",
    ]

    X_example = vectorizer.transform(examples)

    example_pred = clf.predict(X_example)
    example_prob = clf.predict_proba(X_example)

    print()
    print("EXAMPLE CLASSIFICATION")
    print("-" * 78)

    for q, pred, prob in zip(
        examples,
        example_pred,
        example_prob,
    ):
        conf = prob.max()

        print(
            f"{pred:<30} "
            f"conf={conf:.3f}  "
            f"{q}"
        )

    print()
    print("=" * 78)
    print("SAVED")
    print("=" * 78)

    print(f"Classifier      : {output}")
    print(f"Test predictions: {pred_path}")
    print(f"Per-class result: {class_path}")
    print(f"Distribution    : {dist_path}")


if __name__ == "__main__":
    main()
