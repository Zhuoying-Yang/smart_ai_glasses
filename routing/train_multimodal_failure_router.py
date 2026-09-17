import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

from scipy.sparse import csr_matrix, hstack

from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)


# ============================================================
# Configuration
# ============================================================

SIGLIP_MODEL = "google/siglip-base-patch16-224"


# ============================================================
# Split by image_id
# Prevent same image appearing in train and test.
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
# Threshold:
# catch >= target recall of E2B failures,
# then minimize LARGE usage.
# ============================================================

def choose_threshold(y_true, prob, target_recall=0.85):
    results = []

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

        results.append({
            "threshold": float(threshold),
            "recall": float(recall),
            "precision": float(precision),
            "large_rate": float(large_rate),
        })

    feasible = [
        r for r in results
        if r["recall"] >= target_recall
    ]

    if feasible:
        return min(
            feasible,
            key=lambda r: (
                r["large_rate"],
                -r["precision"],
            ),
        )

    return max(
        results,
        key=lambda r: (
            r["recall"],
            -r["large_rate"],
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

    failure_recall = tp / failures if failures else 0
    over_escalation = fp / successes if successes else 0
    small_failure_rate = fn / small_n if small_n else 0

    metrics = {
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(
            y, pred, zero_division=0
        ),
        "recall": recall_score(
            y, pred, zero_division=0
        ),
        "f1": f1_score(
            y, pred, zero_division=0
        ),
        "large_rate": large_n / total,
        "small_rate": small_n / total,
        "over_escalation_rate": over_escalation,
        "small_failure_rate": small_failure_rate,
    }

    print()
    print("=" * 78)
    print(name)
    print("=" * 78)

    print(f"Cases                    : {total:,}")
    print(f"Actual successes         : {successes:,}")
    print(f"Actual failures          : {failures:,}")
    print()

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

    print()
    print(
        f"Failures caught          : "
        f"{tp}/{failures} ({failure_recall*100:.2f}%)"
    )
    print(
        f"Failures missed          : "
        f"{fn}/{failures}"
    )
    print(
        f"Successful cases unnecessarily escalated : "
        f"{fp}/{successes} ({over_escalation*100:.2f}%)"
    )
    print(
        f"Failure rate among requests kept SMALL   : "
        f"{small_failure_rate*100:.2f}%"
    )

    print()
    print("                         Pred SMALL   Pred LARGE")
    print(f"Actual success           {tn:>10}   {fp:>10}")
    print(f"Actual failure           {fn:>10}   {tp:>10}")

    return metrics


# ============================================================
# TF-IDF text features
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
# Device
# ============================================================

def choose_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# Get SigLIP image embeddings
#
# Cached, so this only needs to be done once.
# ============================================================

def build_siglip_cache(
    df,
    cache_path,
    batch_size=32,
):

    cache_path = Path(cache_path)

    # --------------------------------------------------------
    # Already cached
    # --------------------------------------------------------

    if cache_path.exists():

        print()
        print(f"Loading cached SigLIP embeddings:")
        print(cache_path)

        data = np.load(
            cache_path,
            allow_pickle=False,
        )

        image_ids = data["image_ids"]
        embeddings = data["embeddings"]

        return {
            int(i): embeddings[j]
            for j, i in enumerate(image_ids)
        }

    # --------------------------------------------------------
    # Need to compute
    # --------------------------------------------------------

    target_ids = set(
        df["image_id"].astype(int).tolist()
    )

    print()
    print("=" * 78)
    print("SIGLIP IMAGE EMBEDDING")
    print("=" * 78)

    print(
        f"Unique target images : "
        f"{len(target_ids):,}"
    )

    print("Loading cached VQAv2 validation dataset...")

    dataset = load_dataset(
        "parquet",
        data_files={
            "validation":
                "hf://datasets/lmms-lab/VQAv2/"
                "data/validation-*.parquet"
        },
        split="validation",
    )

    # --------------------------------------------------------
    # Find one row for each image
    # --------------------------------------------------------

    print("Finding VQAv2 rows for benchmark images...")

    image_to_index = {}

    all_image_ids = dataset["image_id"]

    for idx, image_id in enumerate(
        tqdm(
            all_image_ids,
            desc="Indexing images",
        )
    ):

        image_id = int(image_id)

        if (
            image_id in target_ids
            and image_id not in image_to_index
        ):
            image_to_index[image_id] = idx

        if len(image_to_index) == len(target_ids):
            break

    missing = target_ids - set(image_to_index)

    if missing:
        raise RuntimeError(
            f"Could not find {len(missing)} images."
        )

    print(
        f"Found all {len(image_to_index):,} unique images."
    )

    # --------------------------------------------------------
    # SigLIP
    # --------------------------------------------------------

    device = choose_device()

    print(f"Device            : {device}")
    print(f"Model             : {SIGLIP_MODEL}")

    processor = AutoProcessor.from_pretrained(
        SIGLIP_MODEL
    )

    model = SiglipVisionModel.from_pretrained(
        SIGLIP_MODEL
    )

    model = model.to(device)
    model.eval()

    sorted_ids = sorted(target_ids)

    all_embeddings = []

    print()
    print("Computing image embeddings...")

    for start in tqdm(
        range(0, len(sorted_ids), batch_size),
        desc="SigLIP",
    ):

        batch_ids = sorted_ids[
            start:start + batch_size
        ]

        images = []

        for image_id in batch_ids:

            row_idx = image_to_index[image_id]

            image = dataset[
                row_idx
            ]["image"].convert("RGB")

            images.append(image)

        inputs = processor(
            images=images,
            return_tensors="pt",
        )

        pixel_values = inputs[
            "pixel_values"
        ].to(device)

        with torch.inference_mode():

            output = model(
                pixel_values=pixel_values
            )

            embedding = output.pooler_output

            # L2 normalize
            embedding = embedding / (
                embedding.norm(
                    dim=1,
                    keepdim=True,
                ) + 1e-12
            )

        all_embeddings.append(
            embedding
            .detach()
            .cpu()
            .float()
            .numpy()
        )

    embeddings = np.concatenate(
        all_embeddings,
        axis=0,
    )

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        cache_path,
        image_ids=np.array(
            sorted_ids,
            dtype=np.int64,
        ),
        embeddings=embeddings.astype(
            np.float32
        ),
    )

    print()
    print("Saved SigLIP cache:")
    print(cache_path)

    return {
        int(image_id): embeddings[i]
        for i, image_id in enumerate(sorted_ids)
    }


# ============================================================
# Create image feature matrix
# ============================================================

def image_matrix(df, embedding_map):

    return np.vstack([
        embedding_map[int(image_id)]
        for image_id in df["image_id"]
    ]).astype(np.float32)


# ============================================================
# Logistic regression
# ============================================================

def train_lr(X, y, seed):

    model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver="liblinear",
        class_weight="balanced",
        random_state=seed,
    )

    model.fit(X, y)

    return model


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
        "--cache",
        default=(
            "benchmark/results/final_10k/"
            "siglip_image_embeddings.npz"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "routing/"
            "e2b_multimodal_failure_router.joblib"
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

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    df = pd.read_csv(args.input)

    df = df.dropna(
        subset=[
            "question",
            "image_id",
            "vqa_score",
        ]
    ).copy()

    # Same failure definition as v1
    df["failure"] = (
        df["vqa_score"].astype(float) < 0.5
    ).astype(int)

    print("=" * 78)
    print("MULTIMODAL E2B FAILURE ROUTER")
    print("=" * 78)

    print(f"Cases       : {len(df):,}")
    print(
        f"Successes   : "
        f"{(df.failure == 0).sum():,}"
    )
    print(
        f"Failures    : "
        f"{(df.failure == 1).sum():,}"
    )

    # --------------------------------------------------------
    # Same image-level split
    # --------------------------------------------------------

    train_df, val_df, test_df = group_split(
        df,
        seed=args.seed,
    )

    print()
    print("Split")
    print("-" * 78)
    print(f"Train : {len(train_df):,}")
    print(f"Val   : {len(val_df):,}")
    print(f"Test  : {len(test_df):,}")

    # --------------------------------------------------------
    # Image embeddings
    # --------------------------------------------------------

    embedding_map = build_siglip_cache(
        df=df,
        cache_path=args.cache,
        batch_size=args.batch_size,
    )

    # --------------------------------------------------------
    # Text features
    # --------------------------------------------------------

    print()
    print("Building TF-IDF text features...")

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

    # --------------------------------------------------------
    # Image features
    # --------------------------------------------------------

    X_train_img = image_matrix(
        train_df,
        embedding_map,
    )

    X_val_img = image_matrix(
        val_df,
        embedding_map,
    )

    X_test_img = image_matrix(
        test_df,
        embedding_map,
    )

    y_train = train_df["failure"].values
    y_val = val_df["failure"].values
    y_test = test_df["failure"].values

    # ========================================================
    # 1. TEXT-ONLY BASELINE
    # ========================================================

    print()
    print("Training text-only baseline...")

    text_model = train_lr(
        X_train_text,
        y_train,
        args.seed,
    )

    val_prob_text = text_model.predict_proba(
        X_val_text
    )[:, 1]

    text_threshold_info = choose_threshold(
        y_val,
        val_prob_text,
        args.target_failure_recall,
    )

    text_threshold = text_threshold_info[
        "threshold"
    ]

    test_prob_text = text_model.predict_proba(
        X_test_text
    )[:, 1]

    text_metrics = evaluate(
        "TEXT-ONLY HELD-OUT TEST",
        y_test,
        test_prob_text,
        text_threshold,
    )

    # ========================================================
    # 2. MULTIMODAL
    # ========================================================

    print()
    print("Combining text + SigLIP image features...")

    X_train_mm = hstack([
        X_train_text,
        csr_matrix(X_train_img),
    ]).tocsr()

    X_val_mm = hstack([
        X_val_text,
        csr_matrix(X_val_img),
    ]).tocsr()

    X_test_mm = hstack([
        X_test_text,
        csr_matrix(X_test_img),
    ]).tocsr()

    print("Training multimodal router...")

    mm_model = train_lr(
        X_train_mm,
        y_train,
        args.seed,
    )

    val_prob_mm = mm_model.predict_proba(
        X_val_mm
    )[:, 1]

    mm_threshold_info = choose_threshold(
        y_val,
        val_prob_mm,
        args.target_failure_recall,
    )

    mm_threshold = mm_threshold_info[
        "threshold"
    ]

    test_prob_mm = mm_model.predict_proba(
        X_test_mm
    )[:, 1]

    mm_metrics = evaluate(
        "MULTIMODAL HELD-OUT TEST",
        y_test,
        test_prob_mm,
        mm_threshold,
    )

    # ========================================================
    # Direct comparison
    # ========================================================

    print()
    print("=" * 78)
    print("TEXT-ONLY vs MULTIMODAL")
    print("=" * 78)

    print(
        f"{'Metric':<30}"
        f"{'Text':>12}"
        f"{'Multimodal':>14}"
        f"{'Delta':>12}"
    )

    print("-" * 78)

    comparison = [
        (
            "ROC-AUC",
            text_metrics["roc_auc"],
            mm_metrics["roc_auc"],
        ),
        (
            "PR-AUC",
            text_metrics["pr_auc"],
            mm_metrics["pr_auc"],
        ),
        (
            "Failure recall",
            text_metrics["recall"],
            mm_metrics["recall"],
        ),
        (
            "LARGE routing rate",
            text_metrics["large_rate"],
            mm_metrics["large_rate"],
        ),
        (
            "Over-escalation",
            text_metrics[
                "over_escalation_rate"
            ],
            mm_metrics[
                "over_escalation_rate"
            ],
        ),
        (
            "SMALL failure rate",
            text_metrics[
                "small_failure_rate"
            ],
            mm_metrics[
                "small_failure_rate"
            ],
        ),
    ]

    for name, a, b in comparison:

        print(
            f"{name:<30}"
            f"{a:>12.4f}"
            f"{b:>14.4f}"
            f"{b-a:>+12.4f}"
        )

    # ========================================================
    # Save model
    # ========================================================

    bundle = {
        "text_encoder": text_encoder,
        "model": mm_model,
        "threshold": float(mm_threshold),
        "siglip_model": SIGLIP_MODEL,
        "failure_definition": "vqa_score < 0.5",
        "target_failure_recall":
            args.target_failure_recall,
        "seed": args.seed,
        "text_only_metrics": text_metrics,
        "multimodal_metrics": mm_metrics,
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

    # --------------------------------------------------------
    # Save held-out predictions
    # --------------------------------------------------------

    pred_df = test_df.copy()

    pred_df[
        "text_failure_probability"
    ] = test_prob_text

    pred_df[
        "multimodal_failure_probability"
    ] = test_prob_mm

    pred_df["text_route"] = np.where(
        test_prob_text >= text_threshold,
        "LARGE",
        "SMALL",
    )

    pred_df["multimodal_route"] = np.where(
        test_prob_mm >= mm_threshold,
        "LARGE",
        "SMALL",
    )

    prediction_path = Path(
        "benchmark/results/final_10k/"
        "multimodal_router_test_predictions.csv"
    )

    pred_df.to_csv(
        prediction_path,
        index=False,
    )

    print()
    print("=" * 78)
    print("SAVED")
    print("=" * 78)

    print(f"Router      : {output_path}")
    print(f"SigLIP cache: {args.cache}")
    print(f"Predictions : {prediction_path}")


if __name__ == "__main__":
    main()
