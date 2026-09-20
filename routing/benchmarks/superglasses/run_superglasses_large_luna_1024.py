from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
from openai import OpenAI
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("routing/benchmarks/superglasses")

INPUT = (
    ROOT
    / "results"
    / "superglasses_large_luna_medium.csv"
)

OUT_JSONL = (
    ROOT
    / "results"
    / "superglasses_large_luna_medium_1024.jsonl"
)

OUT_CSV = (
    ROOT
    / "results"
    / "superglasses_large_luna_medium_1024.csv"
)

MODEL = "gpt-5.6-luna"
REASONING = "medium"

MAX_EDGE = 1024

# Current Luna pricing, just for estimated cost reporting.
INPUT_USD_PER_M = 0.20
OUTPUT_USD_PER_M = 1.20


if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY is not set."
    )


# IMPORTANT:
# max_retries=0 so latency does not secretly include
# automatic SDK retry/backoff.
client = OpenAI(
    timeout=120,
    max_retries=0,
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def resized_data_url(path: Path):
    """
    Resize while preserving aspect ratio.
    Long edge <= 1024 px.
    """
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")

        original_size = im.size

        im.thumbnail(
            (MAX_EDGE, MAX_EDGE),
            Image.Resampling.LANCZOS,
        )

        resized_size = im.size

        buf = io.BytesIO()

        im.save(
            buf,
            format="JPEG",
            quality=90,
        )

    encoded = base64.b64encode(
        buf.getvalue()
    ).decode("utf-8")

    return (
        f"data:image/jpeg;base64,{encoded}",
        original_size,
        resized_size,
    )


# ============================================================
# PROMPT
# ============================================================

def make_prompt(question):
    # Keep identical to previous LARGE benchmark.
    return (
        f"{question}\n\n"
        "Answer the question based on the image. "
        "Answer directly and concisely. "
        "Do not add unnecessary explanation."
    )


# ============================================================
# RESUME SUPPORT
# ============================================================

def load_completed():
    done = {}

    if not OUT_JSONL.exists():
        return done

    with OUT_JSONL.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            try:
                x = json.loads(line)
            except Exception:
                continue

            sid = str(
                x.get("sample_id")
            )

            if (
                x.get("status") == "ok"
                and x.get("large_answer")
            ):
                done[sid] = x

    return done


# ============================================================
# INFERENCE
# ============================================================

def run_one(row):
    image_path = Path(
        row["image_path"]
    )

    if not image_path.exists():
        raise FileNotFoundError(
            image_path
        )

    (
        image_url,
        original_size,
        resized_size,
    ) = resized_data_url(
        image_path
    )

    # Timer starts immediately before API request.
    # Local resize is intentionally NOT included in model latency.
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
                        "type":
                            "input_text",

                        "text":
                            make_prompt(
                                row[
                                    "question"
                                ]
                            ),
                    },

                    {
                        "type":
                            "input_image",

                        "image_url":
                            image_url,

                        "detail":
                            "auto",
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

    usage = response.usage

    input_tokens = int(
        getattr(
            usage,
            "input_tokens",
            0,
        )
        or 0
    )

    output_tokens = int(
        getattr(
            usage,
            "output_tokens",
            0,
        )
        or 0
    )

    estimated_cost = (
        input_tokens
        * INPUT_USD_PER_M
        / 1_000_000
        +
        output_tokens
        * OUTPUT_USD_PER_M
        / 1_000_000
    )

    return {
        "sample_id":
            str(
                row["sample_id"]
            ),

        "question":
            row["question"],

        "ground_truth":
            row["ground_truth"],

        "image_path":
            str(image_path),

        "original_width":
            original_size[0],

        "original_height":
            original_size[1],

        "resized_width":
            resized_size[0],

        "resized_height":
            resized_size[1],

        "large_answer":
            response.output_text.strip(),

        "latency_sec":
            round(
                latency,
                4,
            ),

        "input_tokens":
            input_tokens,

        "output_tokens":
            output_tokens,

        "estimated_cost_usd":
            estimated_cost,

        "model":
            MODEL,

        "reasoning_effort":
            REASONING,

        "max_image_edge":
            MAX_EDGE,

        "status":
            "ok",

        "error":
            None,
    }


# ============================================================
# MAIN
# ============================================================

df = pd.read_csv(
    INPUT
)

# Only use the 964 cases that previously succeeded.
if "status" in df.columns:
    df = df[
        df["status"] == "ok"
    ].copy()

df["sample_id"] = (
    df["sample_id"]
    .astype(str)
)


print("=" * 100)
print("SUPERGLASSES LARGE — LUNA MEDIUM — RESIZE 1024")
print("=" * 100)

print("Model            :", MODEL)
print("Reasoning        :", REASONING)
print("Max image edge   :", MAX_EDGE)
print("Cases            :", len(df))
print("Concurrency      : 1")
print("SDK auto retries : 0")


completed = load_completed()

print(
    "Already completed:",
    len(completed),
)

remaining = df[
    ~df["sample_id"].isin(
        completed.keys()
    )
].copy()

print(
    "Remaining        :",
    len(remaining),
)

print()


with OUT_JSONL.open(
    "a",
    encoding="utf-8",
) as fout:

    for _, row in tqdm(
        remaining.iterrows(),
        total=len(remaining),
        desc="Luna-medium 1024",
    ):

        try:
            record = run_one(
                row
            )

        except Exception as e:
            record = {
                "sample_id":
                    str(
                        row[
                            "sample_id"
                        ]
                    ),

                "question":
                    row.get(
                        "question"
                    ),

                "ground_truth":
                    row.get(
                        "ground_truth"
                    ),

                "image_path":
                    row.get(
                        "image_path"
                    ),

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

                "max_image_edge":
                    MAX_EDGE,

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
        subset=[
            "sample_id"
        ],
        keep="last",
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
].copy()

bad = out[
    out["status"] != "ok"
].copy()


print()
print("=" * 100)
print("RESULT")
print("=" * 100)

print(
    "Attempted          :",
    len(out),
)

print(
    "Successful         :",
    len(good),
)

print(
    "Errors             :",
    len(bad),
)


if len(good):

    latency = pd.to_numeric(
        good[
            "latency_sec"
        ],
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
            latency.quantile(
                0.95
            ),
            3,
        ),
    )

    print(
        "Mean input tokens :",
        round(
            good[
                "input_tokens"
            ]
            .mean(),
            1,
        ),
    )

    print(
        "Total input tokens:",
        int(
            good[
                "input_tokens"
            ]
            .sum()
        ),
    )

    print(
        "Total output tokens:",
        int(
            good[
                "output_tokens"
            ]
            .sum()
        ),
    )

    print(
        "Approx API cost   : $",
        round(
            good[
                "estimated_cost_usd"
            ]
            .sum(),
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
print("CSV:")
print(OUT_CSV)

print()
print("JSONL:")
print(OUT_JSONL)
