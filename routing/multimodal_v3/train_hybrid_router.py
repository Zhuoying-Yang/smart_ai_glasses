import csv
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score

from routing.task_classifier import classify_task


GRADED = Path(
    "routing/benchmarks/results/"
    "wearvqa_e2b_50_graded.csv"
)

MANIFEST = Path(
    "routing/benchmarks/wearvqa_mini/"
    "manifest.csv"
)

RESULT_OUT = Path(
    "routing/multimodal_v3/results/"
    "hybrid_router_loo_scores.csv"
)

MODEL_OUT = Path(
    "routing/multimodal_v3/"
    "hybrid_router.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

with open(GRADED, newline="", encoding="utf-8") as f:
    graded = list(csv.DictReader(f))

graded = [
    r for r in graded
    if r["manual_correct"] in {"0", "1"}
]


with open(MANIFEST, newline="", encoding="utf-8") as f:
    manifest_rows = list(csv.DictReader(f))


manifest = {
    str(r["sample_id"]): r
    for r in manifest_rows
}


# ============================================================
# HELPERS
# ============================================================

def flag_value(value):
    """
    Convert WearVQA metadata flags to 0/1.

    Handles values such as:
        yes/no
        true/false
        1/0
    """

    if value is None:
        return 0.0

    v = str(value).strip().lower()

    if v in {
        "1", "true", "yes", "y",
        "present", "positive"
    }:
        return 1.0

    return 0.0


VISUAL_FLAG_CANDIDATES = [
    "is_blur",
    "is_low_light",
    "is_occluded",
    "is_cut_off",
    "hand_finger_elements",
    "is_leveling",
    "is_not_zoomed_in",
]


# Only use columns actually present in manifest
manifest_columns = set(
    manifest_rows[0].keys()
)

VISUAL_FLAGS = [
    x for x in VISUAL_FLAG_CANDIDATES
    if x in manifest_columns
]


questions = [
    r["question"]
    for r in graded
]

labels = np.array([
    1 if r["manual_correct"] == "0" else 0
    for r in graded
])


# Predict task type using classifier trained outside mini-50
predicted_tasks = []

for r in graded:

    task, _ = classify_task(
        r["question"]
    )

    predicted_tasks.append(
        task
    )


print()
print("=" * 82)
print("HYBRID WHICH ROUTER V3")
print("=" * 82)

print("Judgeable samples :", len(graded))
print("SMALL failures    :", int(labels.sum()))
print("SMALL successes   :", int((1-labels).sum()))
print("Visual flags      :", VISUAL_FLAGS)


# ============================================================
# EMPIRICAL TASK RISK
#
# CRITICAL:
# Calculated inside each training fold.
# The held-out test label is NEVER used.
# ============================================================

def build_task_risks(
    train_indices,
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

    # Global smoothed fallback
    global_risk = (
        (total_wrong + 1)
        / (total_n + 2)
    )

    risks = {}

    for task, (wrong, n) in stats.items():

        # Laplace smoothing
        risks[task] = (
            (wrong + 1)
            / (n + 2)
        )

    return risks, global_risk


def numeric_features(
    indices,
    task_risks,
    global_risk,
):

    features = []

    for i in indices:

        r = graded[i]

        sid = str(
            r["sample_id"]
        )

        meta = manifest.get(
            sid,
            {}
        )

        task = predicted_tasks[i]

        task_risk = task_risks.get(
            task,
            global_risk,
        )

        row = [
            task_risk,
        ]

        for flag in VISUAL_FLAGS:
            row.append(
                flag_value(
                    meta.get(flag)
                )
            )

        features.append(row)

    return np.asarray(
        features,
        dtype=float,
    )


# ============================================================
# LEAVE-ONE-OUT
# ============================================================

loo = LeaveOneOut()

scores = np.zeros(
    len(graded),
    dtype=float,
)


for train_idx, test_idx in loo.split(
    questions
):

    train_q = [
        questions[i]
        for i in train_idx
    ]

    test_q = [
        questions[i]
        for i in test_idx
    ]

    train_y = labels[
        train_idx
    ]

    # --------------------------------------------------------
    # TEXT FEATURES
    # --------------------------------------------------------

    tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        max_features=3000,
    )

    X_train_text = tfidf.fit_transform(
        train_q
    )

    X_test_text = tfidf.transform(
        test_q
    )

    # --------------------------------------------------------
    # FOLD-SAFE TASK PRIOR
    # --------------------------------------------------------

    task_risks, global_risk = build_task_risks(
        train_idx
    )

    # --------------------------------------------------------
    # TASK + VISUAL FEATURES
    # --------------------------------------------------------

    X_train_num = numeric_features(
        train_idx,
        task_risks,
        global_risk,
    )

    X_test_num = numeric_features(
        test_idx,
        task_risks,
        global_risk,
    )

    X_train = hstack([
        X_train_text,
        csr_matrix(X_train_num),
    ])

    X_test = hstack([
        X_test_text,
        csr_matrix(X_test_num),
    ])

    # --------------------------------------------------------
    # LIGHTWEIGHT ROUTER
    # --------------------------------------------------------

    model = LogisticRegression(
        C=0.5,
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        random_state=40,
    )

    model.fit(
        X_train,
        train_y,
    )

    score = model.predict_proba(
        X_test
    )[0, 1]

    scores[
        test_idx[0]
    ] = score


# ============================================================
# EVALUATION
# ============================================================

auc = roc_auc_score(
    labels,
    scores,
)


print()
print(
    "LOO ROC-AUC       :",
    f"{auc:.3f}",
)


# ============================================================
# SAVE SCORES
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
        "router_score",
        "predicted_task",
    ])

    for r, y, score, task in zip(
        graded,
        labels,
        scores,
        predicted_tasks,
    ):

        writer.writerow([
            r["sample_id"],
            r["category"],
            r["question"],
            int(y),
            f"{score:.6f}",
            task,
        ])


print(
    "Saved scores      :",
    RESULT_OUT,
)

print()
