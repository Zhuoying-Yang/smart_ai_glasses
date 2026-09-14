import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from scipy.sparse import csr_matrix, hstack

from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score

from transformers import CLIPModel, CLIPProcessor

from routing.task_classifier import classify_task


# ============================================================
# PATHS
# ============================================================

GRADED = Path(
    "routing/benchmarks/results/"
    "wearvqa_e2b_50_graded.csv"
)

# Images are intentionally NOT stored in GitHub.
IMAGE_DIR = (
    Path.home()
    / "projectaria_client_sdk_samples"
    / "routing"
    / "benchmarks"
    / "wearvqa_mini"
    / "images"
)

CACHE = Path(
    "routing/clip_v4/cache/"
    "clip_features.npz"
)

RESULT_OUT = Path(
    "routing/clip_v4/results/"
    "clip_router_loo_scores.csv"
)


# ============================================================
# LOAD LABELS
# ============================================================

with open(
    GRADED,
    newline="",
    encoding="utf-8",
) as f:
    rows = list(csv.DictReader(f))


rows = [
    r for r in rows
    if r["manual_correct"] in {"0", "1"}
]


sample_ids = [
    str(r["sample_id"])
    for r in rows
]

questions = [
    r["question"]
    for r in rows
]

labels = np.array([
    1 if r["manual_correct"] == "0" else 0
    for r in rows
])


print()
print("=" * 86)
print("CLIP MULTIMODAL WHICH ROUTER V4")
print("=" * 86)

print("Judgeable samples :", len(rows))
print("SMALL failures    :", int(labels.sum()))
print("SMALL successes   :", int((1 - labels).sum()))
print("Image directory   :", IMAGE_DIR)


# ============================================================
# CHECK IMAGES
# ============================================================

missing = []

for sid in sample_ids:

    path = IMAGE_DIR / f"{sid}.jpg"

    if not path.exists():
        missing.append(str(path))


if missing:

    print()
    print("Missing images:")

    for x in missing[:10]:
        print(" ", x)

    raise FileNotFoundError(
        f"{len(missing)} WearVQA images are missing."
    )


# ============================================================
# FROZEN CLIP FEATURES
#
# These features use NO E2B correctness labels,
# so extracting them before cross-validation is safe.
# ============================================================

def compute_clip_features():

    print()
    print("Loading CLIP...")

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Device:", device)

    model_name = (
        "openai/clip-vit-base-patch32"
    )

    processor = CLIPProcessor.from_pretrained(
        model_name
    )

    model = CLIPModel.from_pretrained(
        model_name
    )

    model = model.to(device)
    model.eval()


    all_image_features = []
    all_text_features = []

    batch_size = 8


    for start in range(
        0,
        len(rows),
        batch_size,
    ):

        end = min(
            start + batch_size,
            len(rows),
        )

        batch_rows = rows[start:end]

        images = []

        texts = []


        for r in batch_rows:

            sid = str(
                r["sample_id"]
            )

            image_path = (
                IMAGE_DIR
                / f"{sid}.jpg"
            )

            with Image.open(
                image_path
            ) as img:

                images.append(
                    img.convert("RGB")
                )

            texts.append(
                r["question"]
            )


        image_inputs = processor(
            images=images,
            return_tensors="pt",
        )

        text_inputs = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )


        image_inputs = {
            k: v.to(device)
            for k, v
            in image_inputs.items()
        }

        text_inputs = {
            k: v.to(device)
            for k, v
            in text_inputs.items()
        }


        with torch.inference_mode():

            image_features = (
                model.get_image_features(
                    **image_inputs
                )
            )

            text_features = (
                model.get_text_features(
                    **text_inputs
                )
            )


        image_features = F.normalize(
            image_features,
            dim=-1,
        )

        text_features = F.normalize(
            text_features,
            dim=-1,
        )


        all_image_features.append(
            image_features.cpu().numpy()
        )

        all_text_features.append(
            text_features.cpu().numpy()
        )


        print(
            f"Encoded {end}/{len(rows)}"
        )


    image_features = np.vstack(
        all_image_features
    )

    text_features = np.vstack(
        all_text_features
    )


    CACHE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    np.savez(
        CACHE,
        sample_ids=np.array(
            sample_ids
        ),
        image_features=image_features,
        text_features=text_features,
    )


    return (
        image_features,
        text_features,
    )


# ============================================================
# LOAD OR CREATE CACHE
# ============================================================

use_cache = False


if CACHE.exists():

    cached = np.load(
        CACHE,
        allow_pickle=True,
    )

    cached_ids = [
        str(x)
        for x in cached["sample_ids"]
    ]

    if cached_ids == sample_ids:

        print()
        print(
            "Using cached CLIP features:",
            CACHE,
        )

        image_features = (
            cached["image_features"]
        )

        clip_text_features = (
            cached["text_features"]
        )

        use_cache = True


if not use_cache:

    (
        image_features,
        clip_text_features,
    ) = compute_clip_features()


print()
print(
    "Image feature shape:",
    image_features.shape,
)

print(
    "Text feature shape :",
    clip_text_features.shape,
)


# ============================================================
# CLIP CROSS-MODAL INTERACTION
# ============================================================

# Element-wise interaction between image and
# CLIP text embeddings.
interaction_features = (
    image_features
    * clip_text_features
)

# Since both embeddings are normalized,
# dot product = cosine similarity.
clip_similarity = np.sum(
    image_features
    * clip_text_features,
    axis=1,
    keepdims=True,
)


# ============================================================
# PREDICT TASK TYPES
#
# The task classifier was trained on the other
# WearVQA questions, not these mini-50 samples.
# ============================================================

predicted_tasks = []

for question in questions:

    task, _ = classify_task(
        question
    )

    predicted_tasks.append(
        task
    )


# ============================================================
# FOLD-SAFE TASK PRIOR
#
# IMPORTANT:
# E2B failure labels from the held-out sample
# are NEVER used to calculate its task prior.
# ============================================================

def task_risk_for_fold(
    train_indices,
    requested_indices,
):

    stats = defaultdict(
        lambda: [0, 0]
    )

    total_wrong = 0
    total_n = 0


    for i in train_indices:

        task = predicted_tasks[i]

        y = int(labels[i])

        stats[task][0] += y
        stats[task][1] += 1

        total_wrong += y
        total_n += 1


    global_risk = (
        (total_wrong + 1)
        / (total_n + 2)
    )


    output = []


    for i in requested_indices:

        task = predicted_tasks[i]

        if task in stats:

            wrong, n = stats[task]

            risk = (
                (wrong + 1)
                / (n + 2)
            )

        else:

            risk = global_risk


        output.append([
            risk
        ])


    return np.asarray(
        output,
        dtype=float,
    )


# ============================================================
# TEXT FEATURES
# ============================================================

def make_text_vectorizer():

    return FeatureUnion([

        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
            ),
        ),

        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                sublinear_tf=True,
                max_features=5000,
            ),
        ),

    ])


# ============================================================
# LIGHTWEIGHT ROUTING HEAD
# ============================================================

def fit_and_score(
    X_train,
    y_train,
    X_test,
):

    router = LogisticRegression(
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        max_iter=3000,
        random_state=40,
    )

    router.fit(
        X_train,
        y_train,
    )

    return router.predict_proba(
        X_test
    )[0, 1]


# ============================================================
# LEAVE-ONE-OUT ABLATION
# ============================================================

loo = LeaveOneOut()


text_scores = np.zeros(
    len(rows)
)

image_scores = np.zeros(
    len(rows)
)

clip_mm_scores = np.zeros(
    len(rows)
)

hybrid_scores = np.zeros(
    len(rows)
)


for fold, (
    train_idx,
    test_idx,
) in enumerate(
    loo.split(questions),
    start=1,
):

    train_questions = [
        questions[i]
        for i in train_idx
    ]

    test_questions = [
        questions[i]
        for i in test_idx
    ]

    train_y = labels[
        train_idx
    ]


    # --------------------------------------------------------
    # A. TEXT ONLY
    # --------------------------------------------------------

    vectorizer = make_text_vectorizer()

    X_train_text = (
        vectorizer.fit_transform(
            train_questions
        )
    )

    X_test_text = (
        vectorizer.transform(
            test_questions
        )
    )


    text_scores[
        test_idx[0]
    ] = fit_and_score(
        X_train_text,
        train_y,
        X_test_text,
    )


    # --------------------------------------------------------
    # B. IMAGE ONLY
    # --------------------------------------------------------

    X_train_image = (
        image_features[
            train_idx
        ]
    )

    X_test_image = (
        image_features[
            test_idx
        ]
    )


    image_scores[
        test_idx[0]
    ] = fit_and_score(
        X_train_image,
        train_y,
        X_test_image,
    )


    # --------------------------------------------------------
    # C. CLIP MULTIMODAL
    #
    # image embedding
    # + CLIP question embedding
    # + element-wise interaction
    # + cosine similarity
    # --------------------------------------------------------

    train_mm = np.hstack([
        image_features[
            train_idx
        ],
        clip_text_features[
            train_idx
        ],
        interaction_features[
            train_idx
        ],
        clip_similarity[
            train_idx
        ],
    ])

    test_mm = np.hstack([
        image_features[
            test_idx
        ],
        clip_text_features[
            test_idx
        ],
        interaction_features[
            test_idx
        ],
        clip_similarity[
            test_idx
        ],
    ])


    clip_mm_scores[
        test_idx[0]
    ] = fit_and_score(
        train_mm,
        train_y,
        test_mm,
    )


    # --------------------------------------------------------
    # D. FULL HYBRID
    #
    # TF-IDF request features
    # + CLIP multimodal features
    # + fold-safe task failure prior
    # --------------------------------------------------------

    train_task_risk = (
        task_risk_for_fold(
            train_idx,
            train_idx,
        )
    )

    test_task_risk = (
        task_risk_for_fold(
            train_idx,
            test_idx,
        )
    )


    X_train_hybrid = hstack([
        X_train_text,
        csr_matrix(
            train_mm
        ),
        csr_matrix(
            train_task_risk
        ),
    ])


    X_test_hybrid = hstack([
        X_test_text,
        csr_matrix(
            test_mm
        ),
        csr_matrix(
            test_task_risk
        ),
    ])


    hybrid_scores[
        test_idx[0]
    ] = fit_and_score(
        X_train_hybrid,
        train_y,
        X_test_hybrid,
    )


    if (
        fold % 10 == 0
        or fold == len(rows)
    ):

        print(
            f"LOO fold "
            f"{fold}/{len(rows)}"
        )


# ============================================================
# RESULTS
# ============================================================

text_auc = roc_auc_score(
    labels,
    text_scores,
)

image_auc = roc_auc_score(
    labels,
    image_scores,
)

clip_mm_auc = roc_auc_score(
    labels,
    clip_mm_scores,
)

hybrid_auc = roc_auc_score(
    labels,
    hybrid_scores,
)


print()
print("=" * 86)
print("V4 LOO ROC-AUC RESULTS")
print("=" * 86)

print(
    f"Text only                 : "
    f"{text_auc:.3f}"
)

print(
    f"CLIP image only           : "
    f"{image_auc:.3f}"
)

print(
    f"CLIP image + question     : "
    f"{clip_mm_auc:.3f}"
)

print(
    f"Full hybrid + task prior  : "
    f"{hybrid_auc:.3f}"
)


# ============================================================
# SAVE ALL OUT-OF-SAMPLE SCORES
# ============================================================

RESULT_OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with open(
    RESULT_OUT,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "sample_id",
        "category",
        "question",
        "small_failed",
        "predicted_task",
        "text_score",
        "image_score",
        "clip_multimodal_score",
        "hybrid_score",
    ])


    for (
        r,
        y,
        task,
        ts,
        ims,
        mms,
        hs,
    ) in zip(
        rows,
        labels,
        predicted_tasks,
        text_scores,
        image_scores,
        clip_mm_scores,
        hybrid_scores,
    ):

        writer.writerow([
            r["sample_id"],
            r["category"],
            r["question"],
            int(y),
            task,
            f"{ts:.6f}",
            f"{ims:.6f}",
            f"{mms:.6f}",
            f"{hs:.6f}",
        ])


print()
print(
    "Saved:",
    RESULT_OUT,
)

print()
