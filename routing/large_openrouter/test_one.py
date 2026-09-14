import csv
from pathlib import Path

from routing.large_openrouter.openrouter_large import (
    MODEL,
    ask_openrouter_large,
)


GRADED = Path(
    "routing/benchmarks/results/"
    "wearvqa_e2b_50_graded.csv"
)

IMAGE_DIR = (
    Path.home()
    / "projectaria_client_sdk_samples"
    / "routing"
    / "benchmarks"
    / "wearvqa_mini"
    / "images"
)


with open(
    GRADED,
    newline="",
    encoding="utf-8",
) as f:
    rows = list(csv.DictReader(f))


failures = [
    r for r in rows
    if r.get(
        "manual_correct",
        ""
    ).strip() == "0"
]


r = failures[0]

sample_id = str(
    r["sample_id"]
)

image_path = (
    IMAGE_DIR
    / f"{sample_id}.jpg"
)


print()
print("=" * 80)
print("TEMPORARY LARGE TEST")
print("=" * 80)

print("Model    :", MODEL)
print("ID       :", sample_id)
print("Question :", r["question"])
print("GT       :", r["ground_truth"])

if "e2b_prediction" in r:
    print(
        "E2B      :",
        r["e2b_prediction"]
    )

elif "prediction" in r:
    print(
        "E2B      :",
        r["prediction"]
    )

print("Image    :", image_path)

print()
print("Calling OpenRouter...")


answer, latency_ms = (
    ask_openrouter_large(
        str(image_path),
        r["question"],
    )
)


print()
print("LARGE    :", answer)
print(
    "Latency  :",
    f"{latency_ms / 1000:.3f} s"
)
print()
