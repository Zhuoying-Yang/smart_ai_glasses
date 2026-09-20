from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("routing/benchmarks/superglasses")

MANIFEST = ROOT / "direct_visual_manifest.csv"

SMALL_RESULTS = (
    ROOT
    / "results"
    / "superglasses_direct_visual_e2b.csv"
)

IMAGE_DIR = ROOT / "images"

OUT_JSONL = (
    ROOT
    / "results"
    / "superglasses_large_luna_medium.jsonl"
)

OUT_CSV = (
    ROOT
    / "results"
    / "superglasses_large_luna_medium.csv"
)

MODEL = "gpt-5.6-luna"
REASONING = "medium"

# A few parallel requests are enough.
WORKERS = 6

# Current Luna pricing, only for approximate reporting.
INPUT_USD_PER_M = 0.20
OUTPUT_USD_PER_M = 1.20

OUT_JSONL.parent.mkdir(
    parents=True,
    exist_ok=True,
)

if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY is not set."
    )

client = OpenAI(
    timeout=180,
    max_retries=5,
)


# ============================================================
# HELPERS
# ============================================================

def image_to_data_url(path: Path):
    mime, _ = mimetypes.guess_type(
        str(path)
    )

    if not mime:
        mime = "image/jpeg"

    with path.open("rb") as f:
        b64 = base64.b64encode(
            f.read()
        ).decode("utf-8")

    return f"data:{mime};base64,{b64}"


def make_prompt(question: str):
    return (
        f"{question}\n\n"
        "Answer the question based on the image. "
        "Answer directly and concisely. "
        "Do not add unnecessary explanation."
    )


def get_usage(response):
    usage = getattr(
        response,
        "usage",
        None,
    )

    if usage is None:
        return 0, 0, 0.0

    inp = int(
        getattr(
            usage,
            "input_tokens",
            0,
        )
        or 0
    )

    out = int(
        getattr(
            usage,
            "output_tokens",
            0,
        )
        or 0
    )

    cost = (
        inp * INPUT_USD_PER_M / 1_000_000
        +
        out * OUTPUT_USD_PER_M / 1_000_000
    )

    return inp, out, cost


def completed_ids():
    done = set()

    if not OUT_JSONL.exists():
        return done

    with OUT_JSONL.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue

            if (
                r.get("status") == "ok"
                and r.get("large_answer")
            ):
                done.add(
                    str(r["sample_id"])
                )

    return done


# ============================================================
# LOAD DATA
# ============================================================

manifest = pd.read_csv(
    MANIFEST
)

small = pd.read_csv(
    SMALL_RESULTS
)

manifest["sample_id"] = (
    manifest["sample_id"]
    .astype(str)
)

small["sample_id"] = (
    small["sample_id"]
    .astype(str)
)

# Only keep successful E2B cases.
small_ok = small[
    small["error"].isna()
    &
    small["e2b_answer"].notna()
].copy()

df = small_ok[
    ["sample_id"]
].merge(
    manifest,
    on="sample_id",
    how="left",
)

print("=" * 100)
print("SUPERGLASSES LARGE BENCHMARK")
print("=" * 100)

print("Model             :", MODEL)
print("Reasoning         :", REASONING)
print("Successful SMALL  :", len(small_ok))
print("Cases for LARGE   :", len(df))

if len(df) != 964:
    print(
        "NOTE: expected about 964 successful "
        "SMALL cases, found",
        len(df),
    )


# ============================================================
# BUILD CASES
# ============================================================

cases = []

missing_images = []

for _, r in df.iterrows():

    sid = str(
        r["sample_id"]
    )

    image_id = str(
        r["image_id"]
    )

    # Handle possible pandas "123.0".
    if image_id.endswith(".0"):
        image_id = image_id[:-2]

    image_path = None

    for suffix in [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    ]:
        p = (
            IMAGE_DIR
            / f"{image_id}{suffix}"
        )

        if p.exists():
            image_path = p
            break

    if image_path is None:
        missing_images.append(
            (sid, image_id)
        )
        continue

    cases.append({
        "sample_id": sid,
        "image_id": image_id,
        "image_path": str(
            image_path
        ),
        "question": str(
            r["question"]
        ),
        "ground_truth": str(
            r["ground_truth"]
        ),
    })


print("Resolved images   :", len(cases))
print("Missing images    :", len(missing_images))

if missing_images:
    print(
        "First missing:",
        missing_images[:10],
    )


# ============================================================
# LARGE INFERENCE
# ============================================================

def run_one(case):

    t0 = time.perf_counter()

    response = client.responses.create(
        model=MODEL,

        reasoning={
            "effort": REASONING,
        },

        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": make_prompt(
                            case["question"]
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url":
                            image_to_data_url(
                                Path(
                                    case[
                                        "image_path"
                                    ]
                                )
                            ),
                        "detail": "auto",
                    },
                ],
            }
        ],

        max_output_tokens=1000,
    )

    latency = (
        time.perf_counter()
        - t0
    )

    inp, out, cost = get_usage(
        response
    )

    return {
        **case,

        "large_answer":
            response.output_text.strip(),

        "latency_sec":
            round(
                latency,
                4,
            ),

        "input_tokens":
            inp,

        "output_tokens":
            out,

        "estimated_cost_usd":
            cost,

        "model":
            MODEL,

        "reasoning_effort":
            REASONING,

        "status":
            "ok",

        "error":
            None,
    }


done = completed_ids()

pending = [
    c
    for c in cases
    if c["sample_id"] not in done
]

print()
print("Already completed :", len(done))
print("Remaining         :", len(pending))
print()


with OUT_JSONL.open(
    "a",
    encoding="utf-8",
) as fout:

    with ThreadPoolExecutor(
        max_workers=WORKERS
    ) as executor:

        futures = {
            executor.submit(
                run_one,
                case,
            ): case
            for case in pending
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Luna-medium LARGE",
        ):

            case = futures[
                future
            ]

            try:
                record = (
                    future.result()
                )

            except Exception as e:
                record = {
                    **case,

                    "large_answer":
                        None,

                    "latency_sec":
                        None,

                    "input_tokens":
                        0,

                    "output_tokens":
                        0,

                    "estimated_cost_usd":
                        0,

                    "model":
                        MODEL,

                    "reasoning_effort":
                        REASONING,

                    "status":
                        "error",

                    "error":
                        (
                            f"{type(e).__name__}: "
                            f"{e}"
                        ),
                }

            fout.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            fout.flush()


# ============================================================
# REBUILD FINAL CSV
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
    .sort_values(
        "sample_id"
    )
)


out.to_csv(
    OUT_CSV,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

good = out[
    out["status"] == "ok"
]

bad = out[
    out["status"] != "ok"
]

latency = pd.to_numeric(
    good["latency_sec"],
    errors="coerce",
)

cost = pd.to_numeric(
    good["estimated_cost_usd"],
    errors="coerce",
).fillna(0)


print()
print("=" * 100)
print("SUPERGLASSES LARGE RESULTS")
print("=" * 100)

print("Attempted          :", len(out))
print("Successful         :", len(good))
print("Errors             :", len(bad))

if len(good):
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

    print(
        "Input tokens      :",
        int(
            good[
                "input_tokens"
            ].sum()
        ),
    )

    print(
        "Output tokens     :",
        int(
            good[
                "output_tokens"
            ].sum()
        ),
    )

    print(
        "Approx API cost   : $",
        round(
            cost.sum(),
            2,
        ),
        sep="",
    )


if len(bad):
    print()
    print("First errors:")

    print(
        bad[
            [
                "sample_id",
                "question",
                "error",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


print()
print("JSONL:")
print(OUT_JSONL)

print()
print("CSV:")
print(OUT_CSV)
