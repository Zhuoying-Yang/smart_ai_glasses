from __future__ import annotations

import base64
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
from openai import OpenAI
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

ROOT = Path.cwd()

INPUT = Path(
    "routing/benchmarks/superglasses/results/"
    "superglasses_large_luna_medium_1024.csv"
)

OUT_JSONL = Path(
    "routing/benchmarks/superglasses/results/"
    "superglasses_large_luna_medium_1024_sol_graded.jsonl"
)

OUT_CSV = Path(
    "routing/benchmarks/superglasses/results/"
    "superglasses_large_luna_medium_1024_sol_graded.csv"
)

MODEL = "gpt-5.6-sol"
REASONING = "medium"

MAX_EDGE = 1024
MAX_OUTPUT_TOKENS = 2048

# Grading latency does NOT matter here,
# so concurrency is fine.
WORKERS = 4

LABELS = {
    "CORRECT",
    "PARTIAL",
    "INCORRECT",
    "AMBIGUOUS",
}

# Current Sol pricing for summary only.
INPUT_USD_PER_M = 4.0
OUTPUT_USD_PER_M = 20.0


if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY is not set."
    )


# ============================================================
# PROMPT
# ============================================================

PROMPT = """You are evaluating a LARGE vision-language model used in wearable AI glasses.

You can see the actual image.

Your goal is NOT to require an exact match to the reference answer.
Determine whether the LARGE model's answer is sufficiently correct
and useful for the person wearing the glasses.

Use BOTH the image and the reference answer.

The reference is guidance, but it may be overly specific or describe
a visually ambiguous attribute.

Labels:

CORRECT
- The core information requested is correct.
- Minor omissions and harmless wording differences are okay.

PARTIAL
- The answer is substantially useful and close enough for the user,
  but contains a minor perceptual difference, missing detail, or imprecision.
- This includes visually similar colors, nearby fine-grained categories,
  approximate positions, or omitted non-essential details.
- PARTIAL means the answer remains usable for the wearer.

INCORRECT
- The core requested information is clearly wrong.
- It gives a clearly wrong count, object, action, relation, text,
  conclusion, or answers the wrong aspect of the question.

AMBIGUOUS
- The image itself does not allow a confident decision.
- Both answers appear visually plausible.
- The reference may itself be questionable.

Important:
- Do not mark INCORRECT merely because wording differs.
- If red/pink, green/yellow, or similar attributes are genuinely
  hard to distinguish from the image, prefer PARTIAL or AMBIGUOUS.
- Fine-grained category confusion can be PARTIAL if the answer remains useful.
- Do not use PARTIAL to excuse a clearly wrong core answer.

QUESTION:
{question}

REFERENCE ANSWER:
{ground_truth}

LARGE MODEL ANSWER:
{prediction}

Return exactly one JSON object:

{{"label":"PARTIAL","reason":"short reason"}}

label must be exactly one of:
CORRECT
PARTIAL
INCORRECT
AMBIGUOUS
"""


# ============================================================
# HELPERS
# ============================================================

def resolve_path(value):
    p = Path(str(value)).expanduser()

    if p.is_absolute():
        return p

    return ROOT / p


def get_prediction(row):
    """
    Supports either column naming style.
    """
    for key in [
        "large_answer",
        "prediction",
        "answer",
    ]:
        if (
            key in row
            and pd.notna(row[key])
            and str(row[key]).strip()
        ):
            return str(row[key]).strip()

    return ""


def get_image_path(row):
    for key in [
        "image_path",
        "image",
    ]:
        if (
            key in row
            and pd.notna(row[key])
            and str(row[key]).strip()
        ):
            return resolve_path(
                row[key]
            )

    raise ValueError(
        "No image/image_path column found."
    )


def resize_image_data_url(
    image_path,
    max_edge=1024,
):
    """
    Give Sol the same max resolution used
    by the final Luna configuration.
    """
    with Image.open(image_path) as im:

        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")

        original_w, original_h = im.size

        im.thumbnail(
            (max_edge, max_edge),
            Image.Resampling.LANCZOS,
        )

        resized_w, resized_h = im.size

        buf = io.BytesIO()

        im.save(
            buf,
            format="JPEG",
            quality=90,
        )

    encoded = base64.b64encode(
        buf.getvalue()
    ).decode("ascii")

    return (
        f"data:image/jpeg;base64,{encoded}",
        original_w,
        original_h,
        resized_w,
        resized_h,
    )


def parse_output(text):

    text = (
        text
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    parsed = json.loads(text)

    label = str(
        parsed["label"]
    ).upper().strip()

    if label not in LABELS:
        raise ValueError(
            f"Unexpected label: {label}"
        )

    reason = str(
        parsed.get(
            "reason",
            "",
        )
    ).strip()

    return label, reason


def usage_value(obj, name):
    if obj is None:
        return None

    return getattr(
        obj,
        name,
        None,
    )


# ============================================================
# RESUME
# ============================================================

def load_previous():

    latest = {}

    if not OUT_JSONL.exists():
        return latest

    with OUT_JSONL.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            try:
                x = json.loads(line)

                sid = str(
                    x["sample_id"]
                )

                latest[sid] = x

            except Exception:
                pass

    return latest


# ============================================================
# JUDGE ONE
# ============================================================

def judge_one(row_dict):

    sid = str(
        row_dict["sample_id"]
    )

    prediction = get_prediction(
        row_dict
    )

    if not prediction:

        return {
            "sample_id": sid,
            "judge_label": None,
            "judge_reason": None,
            "judge_error":
                "missing_large_prediction",
        }

    image_path = get_image_path(
        row_dict
    )

    if not image_path.exists():

        return {
            "sample_id": sid,
            "judge_label": None,
            "judge_reason": None,
            "judge_error":
                f"image_missing: {image_path}",
        }

    try:

        (
            image_url,
            original_w,
            original_h,
            resized_w,
            resized_h,
        ) = resize_image_data_url(
            image_path,
            MAX_EDGE,
        )

        prompt = PROMPT.format(
            question=str(
                row_dict[
                    "question"
                ]
            ),
            ground_truth=str(
                row_dict[
                    "ground_truth"
                ]
            ),
            prediction=prediction,
        )

        # Separate client per worker.
        client = OpenAI(
            max_retries=2,
            timeout=120,
        )

        started = time.perf_counter()

        response = client.responses.create(

            model=MODEL,

            reasoning={
                "effort": REASONING,
            },

            max_output_tokens=
                MAX_OUTPUT_TOKENS,

            store=False,

            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type":
                                "input_text",
                            "text":
                                prompt,
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
        )

        latency = (
            time.perf_counter()
            - started
        )

        label, reason = parse_output(
            response.output_text
        )

        usage = getattr(
            response,
            "usage",
            None,
        )

        input_tokens = (
            usage_value(
                usage,
                "input_tokens",
            )
            or 0
        )

        output_tokens = (
            usage_value(
                usage,
                "output_tokens",
            )
            or 0
        )

        return {

            "sample_id":
                sid,

            "question":
                str(
                    row_dict[
                        "question"
                    ]
                ),

            "ground_truth":
                str(
                    row_dict[
                        "ground_truth"
                    ]
                ),

            "large_answer":
                prediction,

            "image_path":
                str(
                    image_path
                ),

            "original_width":
                original_w,

            "original_height":
                original_h,

            "judge_width":
                resized_w,

            "judge_height":
                resized_h,

            "judge_label":
                label,

            "judge_reason":
                reason,

            "usable":
                1
                if label in {
                    "CORRECT",
                    "PARTIAL",
                }
                else (
                    0
                    if label
                    == "INCORRECT"
                    else None
                ),

            "judge_model":
                MODEL,

            "judge_reasoning":
                REASONING,

            "judge_latency_s":
                latency,

            "judge_input_tokens":
                input_tokens,

            "judge_output_tokens":
                output_tokens,

            "judge_error":
                None,
        }

    except Exception as e:

        return {

            "sample_id":
                sid,

            "judge_label":
                None,

            "judge_reason":
                None,

            "usable":
                None,

            "judge_error":
                (
                    f"{type(e).__name__}: "
                    f"{e}"
                ),
        }


# ============================================================
# LOAD LARGE RESULTS
# ============================================================

df = pd.read_csv(
    INPUT
)

if "status" in df.columns:

    df = df[
        df["status"] == "ok"
    ].copy()


df["sample_id"] = (
    df["sample_id"]
    .astype(str)
)


print("=" * 90)
print("SUPERGLASSES LARGE IMAGE-AWARE GRADING")
print("=" * 90)

print(
    "Input          :",
    INPUT,
)

print(
    "Cases          :",
    len(df),
)

print(
    "Judge          :",
    MODEL,
)

print(
    "Reasoning      :",
    REASONING,
)

print(
    "Judge image    :",
    f"long edge <= {MAX_EDGE}",
)

print(
    "Workers        :",
    WORKERS,
)


previous = load_previous()

done_ids = {
    sid
    for sid, x
    in previous.items()
    if x.get(
        "judge_label"
    ) in LABELS
}


remaining_df = df[
    ~df["sample_id"].isin(
        done_ids
    )
].copy()


print(
    "Completed      :",
    len(done_ids),
)

print(
    "Remaining      :",
    len(remaining_df),
)

print()


# ============================================================
# RUN
# ============================================================

OUT_JSONL.parent.mkdir(
    parents=True,
    exist_ok=True,
)


rows = (
    remaining_df
    .to_dict(
        orient="records"
    )
)


with ThreadPoolExecutor(
    max_workers=WORKERS
) as executor:

    future_to_id = {

        executor.submit(
            judge_one,
            row,
        ):
        str(
            row["sample_id"]
        )

        for row in rows
    }

    with OUT_JSONL.open(
        "a",
        encoding="utf-8",
    ) as fout:

        for future in tqdm(
            as_completed(
                future_to_id
            ),
            total=len(
                future_to_id
            ),
            desc="Sol-medium judging",
        ):

            result = future.result()

            fout.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            fout.flush()


# ============================================================
# REBUILD FINAL CSV
# ============================================================

latest = load_previous()

grade_df = pd.DataFrame(
    latest.values()
)

# Join original Luna result columns
# so everything stays in one final file.

drop_cols = [
    c
    for c in grade_df.columns
    if (
        c in df.columns
        and c != "sample_id"
    )
]

grade_for_merge = (
    grade_df.drop(
        columns=drop_cols,
        errors="ignore",
    )
)

final = df.merge(
    grade_for_merge,
    on="sample_id",
    how="left",
)


final.to_csv(
    OUT_CSV,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

valid = final[
    final[
        "judge_label"
    ].isin(
        LABELS
    )
].copy()


print()
print("=" * 90)
print("FINAL GRADING RESULT")
print("=" * 90)

print(
    "Total cases        :",
    len(final),
)

print(
    "Successfully judged:",
    len(valid),
)

print(
    "Errors             :",
    final[
        "judge_label"
    ].isna().sum(),
)


print()
print("Labels:")

print(
    valid[
        "judge_label"
    ]
    .value_counts()
    .to_string()
)


judgeable = valid[
    valid[
        "judge_label"
    ] != "AMBIGUOUS"
].copy()


if len(judgeable):

    usable_rate = (
        judgeable[
            "judge_label"
        ].isin(
            [
                "CORRECT",
                "PARTIAL",
            ]
        )
        .mean()
    )

    strict_accuracy = (
        judgeable[
            "judge_label"
        ]
        == "CORRECT"
    ).mean()

    print()

    print(
        "Usable rate "
        "(CORRECT + PARTIAL):",
        f"{usable_rate*100:.2f}%",
    )

    print(
        "Strict CORRECT rate       :",
        f"{strict_accuracy*100:.2f}%",
    )


input_tokens = (
    pd.to_numeric(
        valid[
            "judge_input_tokens"
        ],
        errors="coerce",
    )
    .fillna(0)
    .sum()
)

output_tokens = (
    pd.to_numeric(
        valid[
            "judge_output_tokens"
        ],
        errors="coerce",
    )
    .fillna(0)
    .sum()
)


estimated_cost = (
    input_tokens
    / 1_000_000
    * INPUT_USD_PER_M
    +
    output_tokens
    / 1_000_000
    * OUTPUT_USD_PER_M
)


print()

print(
    "Judge input tokens :",
    int(input_tokens),
)

print(
    "Judge output tokens:",
    int(output_tokens),
)

print(
    "Approx judge cost   :",
    f"${estimated_cost:.2f}",
)


if "category" in final.columns:

    tmp = final[
        final[
            "judge_label"
        ].isin(
            [
                "CORRECT",
                "PARTIAL",
                "INCORRECT",
            ]
        )
    ].copy()

    tmp["usable_binary"] = (
        tmp[
            "judge_label"
        ].isin(
            [
                "CORRECT",
                "PARTIAL",
            ]
        )
        .astype(int)
    )

    print()
    print("Usable rate by category:")

    category_table = (
        tmp
        .groupby(
            "category"
        )
        ["usable_binary"]
        .agg(
            [
                "count",
                "mean",
            ]
        )
        .sort_values(
            "mean"
        )
    )

    category_table[
        "mean"
    ] *= 100

    print(
        category_table
        .to_string()
    )


print()

print(
    "Saved CSV:",
    OUT_CSV,
)

print(
    "Saved JSONL:",
    OUT_JSONL,
)
