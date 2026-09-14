import csv
import json
import pickle
import tarfile
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score


TAR_PATH = Path(
    "routing/benchmarks/wearvqa_raw/wearvqa.tar"
)

MINI_MANIFEST = Path(
    "routing/benchmarks/wearvqa_mini/manifest.csv"
)

MODEL_PATH = Path(
    "routing/task_classifier.pkl"
)


# =========================================================
# HOLD OUT THE 50 MINI SAMPLES
# =========================================================

with open(
    MINI_MANIFEST,
    newline="",
    encoding="utf-8",
) as f:
    mini_rows = list(csv.DictReader(f))

held_out_ids = {
    row["sample_id"]
    for row in mini_rows
}


# =========================================================
# LOAD ALL WEARVQA QUESTIONS
# =========================================================

train_questions = []
train_labels = []

test_questions = []
test_labels = []
test_ids = []


with tarfile.open(TAR_PATH, "r") as tar:

    for member in tar:

        if not member.name.endswith(".json"):
            continue

        f = tar.extractfile(member)

        if f is None:
            continue

        data = json.load(f)

        sample_id = Path(member.name).stem

        question = data["question"]
        label = data["question_type"]

        if sample_id in held_out_ids:
            test_questions.append(question)
            test_labels.append(label)
            test_ids.append(sample_id)

        else:
            train_questions.append(question)
            train_labels.append(label)


print("=" * 75)
print("WEARVQA TASK CLASSIFIER")
print("=" * 75)

print("Training samples :", len(train_questions))
print("Held-out samples :", len(test_questions))


# =========================================================
# TRAIN
# =========================================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        ),
    ),
    (
        "clf",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=40,
        ),
    ),
])


model.fit(
    train_questions,
    train_labels,
)


# =========================================================
# EVALUATE ON YOUR 50 HELD-OUT QUESTIONS
# =========================================================

pred = model.predict(
    test_questions
)

acc = accuracy_score(
    test_labels,
    pred,
)


print()
print("=" * 75)
print("HELD-OUT MINI-50 CLASSIFICATION")
print("=" * 75)

print(
    f"Accuracy: {acc*100:.1f}%"
)

print()

print(
    classification_report(
        test_labels,
        pred,
        zero_division=0,
    )
)


print("=" * 75)
print("MISCLASSIFICATIONS")
print("=" * 75)

for sid, q, gt, p in zip(
    test_ids,
    test_questions,
    test_labels,
    pred,
):

    if gt != p:
        print()
        print("ID      :", sid)
        print("Question:", q)
        print("GT      :", gt)
        print("Pred    :", p)


# =========================================================
# SAVE
# =========================================================

with open(
    MODEL_PATH,
    "wb",
) as f:
    pickle.dump(model, f)


print()
print("Saved:", MODEL_PATH)
