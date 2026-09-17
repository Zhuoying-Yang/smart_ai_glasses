from pathlib import Path
import sys
import json
import time
import pandas as pd
from tqdm import tqdm


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    ROOT
    / "routing/benchmarks/wearvqa_full/manifest.csv"
)

RESULT_DIR = (
    ROOT
    / "routing/benchmarks/results"
)

JSONL_OUT = (
    RESULT_DIR
    / "wearvqa_e2b_full_raw.jsonl"
)

CSV_OUT = (
    RESULT_DIR
    / "wearvqa_e2b_full_raw.csv"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Import existing Galaxy bridge
# ============================================================

BRIDGE_DIR = (
    Path.home()
    / "projectaria_client_sdk_samples"
)

if not BRIDGE_DIR.exists():
    raise FileNotFoundError(
        f"Galaxy bridge directory not found: {BRIDGE_DIR}"
    )

sys.path.insert(
    0,
    str(BRIDGE_DIR),
)

from phone_bridge import ask_phone


# ============================================================
# Prompt
# ============================================================

def build_prompt(question: str) -> str:
    return (
        f"{question}\n"
        "Answer the question directly and concisely. "
        "Do not add unnecessary explanation."
    )


# ============================================================
# Resume existing run
# ============================================================

completed = {}

if JSONL_OUT.exists():
    with JSONL_OUT.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
                completed[
                    str(row["sample_id"])
                ] = row
            except Exception:
                pass


print("=" * 80)
print("WEARVQA FULL → GALAXY E2B")
print("=" * 80)

print(
    f"Already completed : "
    f"{len(completed):,}"
)


# ============================================================
# Load manifest
# ============================================================

df = pd.read_csv(MANIFEST)

print(
    f"Total cases       : "
    f"{len(df):,}"
)

remaining = (
    len(df) - len(completed)
)

print(
    f"Remaining         : "
    f"{remaining:,}"
)

print()


# ============================================================
# Run every WearVQA case through E2B
# ============================================================

with JSONL_OUT.open(
    "a",
    encoding="utf-8",
) as fout:

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Galaxy E2B WearVQA",
    ):

        sample_id = str(
            row["sample_id"]
        )

        if sample_id in completed:
            continue

        image_path = Path(
            row["image"]
        )

        if not image_path.is_absolute():
            image_path = (
                ROOT / image_path
            )

        question = str(
            row["question"]
        )

        record = {
            "sample_id":
                row["sample_id"],

            "image":
                str(row["image"]),

            "question":
                question,

            "ground_truth":
                str(row["ground_truth"]),

            "category":
                str(row["category"]),

            "domain":
                str(row["domain"]),

            "prediction":
                "",

            "latency_s":
                None,

            "error":
                None,
        }

        try:
            prompt = build_prompt(
                question
            )

            start = time.perf_counter()

            prediction = ask_phone(
                str(image_path),
                prompt,
                timeout=30,
            )

            latency = (
                time.perf_counter()
                - start
            )

            record["prediction"] = str(
                prediction
            ).strip()

            record["latency_s"] = float(
                latency
            )

        except Exception as e:
            record["error"] = (
                f"{type(e).__name__}: {e}"
            )

        fout.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

        fout.flush()

        completed[
            sample_id
        ] = record


# ============================================================
# Build CSV from JSONL
# ============================================================

records = []

with JSONL_OUT.open(
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        try:
            records.append(
                json.loads(line)
            )
        except Exception:
            pass


result = pd.DataFrame(
    records
)

# One record per sample in case of resumed/repeated runs.
result = (
    result
    .drop_duplicates(
        subset=["sample_id"],
        keep="last",
    )
    .reset_index(drop=True)
)

result.to_csv(
    CSV_OUT,
    index=False,
)


# ============================================================
# Summary
# ============================================================

success = result[
    result["error"].isna()
]

errors = result[
    result["error"].notna()
]

print()
print("=" * 80)
print("WEARVQA E2B RUN COMPLETE")
print("=" * 80)

print(
    f"Completed inference : "
    f"{len(result):,}"
)

print(
    f"Successful          : "
    f"{len(success):,}"
)

print(
    f"Errors              : "
    f"{len(errors):,}"
)

if len(success):
    print(
        f"Mean latency        : "
        f"{success['latency_s'].mean():.2f} s"
    )

    print(
        f"Median latency      : "
        f"{success['latency_s'].median():.2f} s"
    )

    print(
        f"P95 latency         : "
        f"{success['latency_s'].quantile(0.95):.2f} s"
    )

print()
print("Raw JSONL :", JSONL_OUT)
print("CSV       :", CSV_OUT)
