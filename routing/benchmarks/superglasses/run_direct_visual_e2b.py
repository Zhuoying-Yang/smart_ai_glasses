from pathlib import Path
import sys
import json
import time

import pandas as pd
from tqdm import tqdm


# ============================================================
# Paths
# ============================================================

ROOT = Path("routing/benchmarks/superglasses")

MANIFEST = ROOT / "direct_visual_manifest.csv"
IMAGE_DIR = ROOT / "images"

OUT_JSONL = (
    ROOT
    / "results"
    / "superglasses_direct_visual_e2b.jsonl"
)

OUT_CSV = (
    ROOT
    / "results"
    / "superglasses_direct_visual_e2b.csv"
)

OUT_JSONL.parent.mkdir(
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

sys.path.insert(
    0,
    str(BRIDGE_DIR),
)

from phone_bridge import ask_phone


# ============================================================
# Configuration
# ============================================================

TIMEOUT = 30

# None = run all 977.
# Change to 900 only if you specifically want exactly 900.
LIMIT = None


# IMPORTANT:
# Keep this prompt consistent with the prompt used for your
# previous WearVQA E2B benchmark if possible.
def make_prompt(question):
    return (
        f"{question}\n"
        "Answer directly and concisely based on the image."
    )


# ============================================================
# Helpers
# ============================================================

def first_present(row, keys):
    for key in keys:
        if key in row.index:
            value = row[key]

            if pd.notna(value):
                value = str(value).strip()

                if value:
                    return value

    return None


def normalize_phone_answer(x):
    if isinstance(x, str):
        return x.strip()

    if isinstance(x, dict):
        for key in [
            "answer",
            "response",
            "text",
            "output",
        ]:
            if key in x:
                return str(x[key]).strip()

    return str(x).strip()


# ============================================================
# Load manifest
# ============================================================

df = pd.read_csv(
    MANIFEST
)

print("=" * 100)
print("SUPERGLASSES DIRECT-VISUAL E2B BENCHMARK")
print("=" * 100)

print("Manifest rows :", len(df))
print("Columns       :", list(df.columns))

if LIMIT is not None:
    df = df.head(LIMIT).copy()

print("Cases to use  :", len(df))


# ============================================================
# Build image lookup
#
# We do this once so we don't repeatedly scan 2357 images.
# ============================================================

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

image_files = [
    p
    for p in IMAGE_DIR.rglob("*")
    if (
        p.is_file()
        and p.suffix.lower()
        in IMAGE_SUFFIXES
    )
]

print("Images found  :", len(image_files))


by_name = {}
by_stem = {}

for p in image_files:
    by_name[p.name] = p

    # Only use first match if duplicate stems exist.
    by_stem.setdefault(
        p.stem,
        p,
    )


def resolve_image(row):

    # First check columns that might already contain a path.
    for key in [
        "image_path",
        "local_image_path",
        "image_file",
        "filename",
        "file_name",
    ]:

        if key not in row.index:
            continue

        value = row[key]

        if pd.isna(value):
            continue

        value = str(value).strip()

        if not value:
            continue

        p = Path(value)

        if p.exists():
            return p

        p2 = IMAGE_DIR / value

        if p2.exists():
            return p2

        if p.name in by_name:
            return by_name[p.name]

        if p.stem in by_stem:
            return by_stem[p.stem]


    # Otherwise resolve by identifiers.
    identifiers = []

    for key in [
        "image_id",
        "id",
        "sample_id",
        "source_id",
        "souce_id",
    ]:

        if key in row.index:
            value = row[key]

            if pd.notna(value):
                identifiers.append(
                    str(value).strip()
                )


    for identifier in identifiers:

        if identifier in by_name:
            return by_name[identifier]

        stem = Path(identifier).stem

        if stem in by_stem:
            return by_stem[stem]


        # Try common extensions.
        for suffix in IMAGE_SUFFIXES:

            name = identifier + suffix

            if name in by_name:
                return by_name[name]


    return None


# ============================================================
# Resume support
# ============================================================

successful = set()

if OUT_JSONL.exists():

    with OUT_JSONL.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            try:
                r = json.loads(line)

                if (
                    r.get("error") is None
                    and r.get("e2b_answer")
                ):
                    successful.add(
                        str(r["sample_id"])
                    )

            except Exception:
                pass


print(
    "Already completed successfully:",
    len(successful),
)


# ============================================================
# Benchmark
# ============================================================

with OUT_JSONL.open(
    "a",
    encoding="utf-8",
) as fout:

    for idx, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Galaxy E2B",
    ):

        sample_id = first_present(
            row,
            [
                "sample_id",
                "id",
                "image_id",
            ],
        )

        if sample_id is None:
            sample_id = str(idx)

        sample_id = str(sample_id)

        if sample_id in successful:
            continue


        question = first_present(
            row,
            ["question"],
        )

        reference_answer = (
            first_present(
                row,
                [
                    "answer",
                    "ground_truth",
                    "reference_answer",
                    "response",
                ],
            )
        )

        image_id = first_present(
            row,
            ["image_id"],
        )

        image_path = resolve_image(
            row
        )


        record = {
            "sample_id":
                sample_id,

            "image_id":
                image_id,

            "question":
                question,

            "reference_answer":
                reference_answer,

            "image_path":
                (
                    str(image_path)
                    if image_path
                    else None
                ),

            "e2b_answer":
                None,

            "latency_sec":
                None,

            "error":
                None,
        }


        # ------------------------------------------
        # Basic validation
        # ------------------------------------------

        if not question:

            record["error"] = (
                "Missing question"
            )

        elif image_path is None:

            record["error"] = (
                "Could not resolve image"
            )

        else:

            # --------------------------------------
            # Galaxy inference
            # --------------------------------------

            try:

                prompt = make_prompt(
                    question
                )

                t0 = time.perf_counter()

                response = ask_phone(
                    str(image_path),
                    prompt,
                    timeout=TIMEOUT,
                )

                latency = (
                    time.perf_counter()
                    - t0
                )

                answer = normalize_phone_answer(
                    response
                )

                record["e2b_answer"] = (
                    answer
                )

                record["latency_sec"] = (
                    round(
                        latency,
                        4,
                    )
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

        # Save every case immediately.
        fout.flush()


# ============================================================
# Rebuild clean CSV using latest result for each sample
# ============================================================

records = []

with OUT_JSONL.open(
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        try:
            records.append(
                json.loads(line)
            )

        except Exception:
            pass


out = pd.DataFrame(
    records
)

out = (
    out
    .drop_duplicates(
        subset=["sample_id"],
        keep="last",
    )
)

out.to_csv(
    OUT_CSV,
    index=False,
)


# ============================================================
# Summary
# ============================================================

success_df = out[
    out["error"].isna()
    & out["e2b_answer"].notna()
]

error_df = out[
    out["error"].notna()
]


print()
print("=" * 100)
print("SUPERGLASSES E2B RESULTS")
print("=" * 100)

print(
    "Attempted          :",
    len(out),
)

print(
    "Successful         :",
    len(success_df),
)

print(
    "Errors             :",
    len(error_df),
)


if len(success_df):

    latency = pd.to_numeric(
        success_df["latency_sec"],
        errors="coerce",
    )

    print(
        "Mean latency (s)  :",
        round(
            latency.mean(),
            3,
        ),
    )

    print(
        "Median latency (s):",
        round(
            latency.median(),
            3,
        ),
    )

    print(
        "P95 latency (s)   :",
        round(
            latency.quantile(0.95),
            3,
        ),
    )


if len(error_df):

    print()
    print("First errors:")

    print(
        error_df[
            [
                "sample_id",
                "question",
                "error",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


print()
print("JSONL:")
print(OUT_JSONL)

print()
print("CSV:")
print(OUT_CSV)
