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

OUT = Path(
    "routing/large_openrouter/results/"
    "openrouter_failure_pilot_5.csv"
)

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with open(GRADED, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# clear E2B failures only
failures = [
    r for r in rows
    if r.get("manual_correct", "").strip() == "0"
]

# first 5 failures for pilot
failures = failures[:5]

done_ids = set()

if OUT.exists():
    with open(OUT, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            done_ids.add(str(r["sample_id"]))


fieldnames = [
    "sample_id",
    "category",
    "question",
    "ground_truth",
    "e2b_prediction",
    "large_prediction",
    "large_latency_ms",
    "status",
]

write_header = not OUT.exists()

with open(OUT, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    if write_header:
        writer.writeheader()

    print()
    print("=" * 80)
    print("OPENROUTER LARGE PILOT")
    print("=" * 80)
    print("Model   :", MODEL)
    print("Samples :", len(failures))
    print("Output  :", OUT)
    print()

    for i, r in enumerate(failures, start=1):
        sample_id = str(r["sample_id"])

        if sample_id in done_ids:
            print(f"[{i}/{len(failures)}] {sample_id} already done, skipping.")
            continue

        image_path = IMAGE_DIR / f"{sample_id}.jpg"

        print("=" * 80)
        print(f"[{i}/{len(failures)}] ID {sample_id}")
        print("Category :", r["category"])
        print("Question :", r["question"])
        print("GT       :", r["ground_truth"])
        print("E2B      :", r["prediction"] if "prediction" in r else r.get("e2b_prediction", ""))
        print("Image    :", image_path)
        print("Calling LARGE...")

        try:
            answer, latency_ms = ask_openrouter_large(
                str(image_path),
                r["question"],
            )
            status = "ok"

            print("LARGE    :", answer)
            print("Latency  :", f"{latency_ms / 1000:.3f} s")

        except Exception as e:
            answer = ""
            latency_ms = ""
            status = f"error: {type(e).__name__}: {e}"

            print("FAILED   :", status)

        writer.writerow({
            "sample_id": sample_id,
            "category": r["category"],
            "question": r["question"],
            "ground_truth": r["ground_truth"],
            "e2b_prediction": r["prediction"] if "prediction" in r else r.get("e2b_prediction", ""),
            "large_prediction": answer,
            "large_latency_ms": latency_ms,
            "status": status,
        })

        f.flush()
        print()

print("Saved:", OUT)
