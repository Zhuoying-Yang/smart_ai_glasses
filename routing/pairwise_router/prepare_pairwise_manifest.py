import csv
from pathlib import Path


SRC = Path(
    "routing/benchmarks/results/"
    "wearvqa_e2b_50_graded.csv"
)

OUT = Path(
    "routing/pairwise_router/"
    "small_large_pairwise.csv"
)


with open(SRC, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))


rows = [
    r for r in rows
    if r["manual_correct"] in {"0", "1"}
]


fieldnames = [
    "sample_id",
    "category",
    "question",

    # SMALL
    "small_correct",

    # Fill these after LARGE is available
    "large_prediction",
    "large_correct",
    "large_latency_ms",

    # Final RouteLLM-style target
    "large_advantage_label",
]


with open(
    OUT,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for r in rows:

        small_correct = int(
            r["manual_correct"] == "1"
        )

        writer.writerow({
            "sample_id": r["sample_id"],
            "category": r["category"],
            "question": r["question"],
            "small_correct": small_correct,
            "large_prediction": "",
            "large_correct": "",
            "large_latency_ms": "",
            "large_advantage_label": "",
        })


print("Saved:", OUT)
print("Samples:", len(rows))
