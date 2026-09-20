import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

OUT = Path(
    "routing/benchmarks/superglasses/full_routing_manifest.csv"
)

def meaningful(x):
    if x is None:
        return False

    if isinstance(x, dict):
        return any(meaningful(v) for v in x.values())

    if isinstance(x, list):
        return any(meaningful(v) for v in x)

    if isinstance(x, str):
        s = x.strip()

        if not s or s.lower() in {
            "none", "null", "{}", "[]"
        }:
            return False

        try:
            return meaningful(json.loads(s))
        except Exception:
            return True

    return bool(x)


def requires_retrieval(row):
    for sub in row.get("sub_questions") or []:
        if meaningful(sub.get("retrieval-tools")):
            return True

    return False


def classify(row):
    categories = set(row.get("category") or [])

    if "Temporal Understanding" in categories:
        return "TEMPORAL"

    if requires_retrieval(row):
        return "KNOWLEDGE"

    return "DIRECT_VISUAL"


ds = load_dataset(
    "xandery/SuperGlasses",
    name="queries",
    split="test",
)

rows = []

for r in ds:
    if r["use_image"] != "Yes":
        continue

    rows.append({
        "sample_id": r["id"],
        "image_id": r["image_id"],
        "image": r["image"],
        "question": r["question"],
        "ground_truth": r["answer"],
        "routing_type": classify(r),
        "categories": "|".join(r["category"] or []),
        "domain": r["domain"],
        "difficulty": r["difficulty"],
        "dynamism": r["dynamism"],
        "image_quality": r["image_quality"],
        "glasses": r["glasses"],
        "location": r["location"],
        "hops": r["hops"],
    })

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

print("=" * 90)
print("FULL SUPERGLASSES MANIFEST")
print("=" * 90)

print("Image questions :", len(df))
print("Unique images   :", df["image_id"].nunique())

print("\nRouting types:")
print(df["routing_type"].value_counts().to_string())

print("\nSaved:")
print(OUT)
