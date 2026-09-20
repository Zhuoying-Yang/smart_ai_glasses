from pathlib import Path

import pandas as pd
from datasets import load_dataset


OUT = Path(
    "routing/benchmarks/supermemory/"
    "question_manifest.csv"
)

ds = load_dataset(
    "OSU-AIoT-MLSys-Lab/SuperMemory-VQA",
    split="test",
)

rows = []

for r in ds:
    meta = r["metadata"]

    rows.append({
        "source": "SuperMemory-VQA",
        "sample_id": r.get(
            "question_id",
            len(rows),
        ),
        "question": r["question"],

        # High-level router label
        "routing_type": "MEMORY",

        # Fine-grained memory subtype
        "subtype": meta["skill"],

        "is_answerable": r["is_answerable"],
        "subject": r["subject"],

        "primary_video_id": meta[
            "primary_video_id"
        ],

        "primary_video_start_time": meta[
            "primary_video_start_time"
        ],
    })


df = pd.DataFrame(rows)

df.to_csv(
    OUT,
    index=False,
)


print("=" * 90)
print("SUPERMEMORY QUESTION MANIFEST")
print("=" * 90)

print("Rows:", len(df))

print()
print("Memory subtype distribution:")
print(
    df["subtype"]
    .value_counts()
    .to_string()
)

print()
print("Answerability:")
print(
    df["is_answerable"]
    .value_counts()
    .to_string()
)

print()
print("Saved:")
print(OUT)
