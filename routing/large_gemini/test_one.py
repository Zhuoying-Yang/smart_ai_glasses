import csv
from pathlib import Path

from routing.large_gemini.gemini_large import (
    MODEL,
    ask_gemini_large,
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


# Only clear E2B failures.
failures = [
    r for r in rows
    if r.get("manual_correct", "").strip() == "0"
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

if "ground_truth" in r:
    print(
        "GT       :",
        r["ground_truth"]
    )

if "prediction" in r:
    print(
        "E2B      :",
        r["prediction"]
    )

elif "e2b_prediction" in r:
    print(
        "E2B      :",
        r["e2b_prediction"]
    )

print("Image    :", image_path)

print()
print("Calling Gemini...")


answer, latency_ms = ask_gemini_large(
    str(image_path),
    r["question"],
)


print()
print("Gemini   :", answer)
print(
    "Latency  :",
    f"{latency_ms / 1000:.3f} s"
)

print()
