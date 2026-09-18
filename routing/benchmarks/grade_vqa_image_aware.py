from __future__ import annotations

import argparse
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


MODEL = "gpt-5.6-sol"
REASONING = "medium"
WORKERS = 4


GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": [
                "CORRECT",
                "PARTIAL",
                "INCORRECT",
                "AMBIGUOUS",
            ],
        },
        "confidence": {
            "type": "number",
        },
        "reason": {
            "type": "string",
        },
    },
    "required": [
        "label",
        "confidence",
        "reason",
    ],
    "additionalProperties": False,
}


def image_to_data_url(path: Path):
    mime, _ = mimetypes.guess_type(str(path))

    if not mime:
        mime = "image/jpeg"

    with path.open("rb") as f:
        data = base64.b64encode(
            f.read()
        ).decode("utf-8")

    return f"data:{mime};base64,{data}"


def make_prompt(
    question,
    ground_truth,
    model_answer,
):
    return f"""
You are grading the answer of a vision-language model for a wearable AI system.

You are given:
1. the original image,
2. the question,
3. a reference / ground-truth answer,
4. the model answer.

Use exactly one label:

CORRECT
- The model answer is substantively correct.
- Equivalent wording is acceptable.

PARTIAL
- The answer is substantially useful and close enough,
  but has a minor perceptual, wording, or specificity difference.
- The main meaning is still useful to the user.
- Example: if the reference says "pink" but the image itself
  makes pink vs. red genuinely difficult to distinguish,
  "red" may be PARTIAL rather than fully INCORRECT.

INCORRECT
- The answer is materially wrong, misleading,
  misses the essential answer, or hallucinates important information.

AMBIGUOUS
- The image, question, or reference does not provide enough
  information to fairly determine correctness,
  or multiple interpretations are genuinely plausible.

Important:
- Look at the ORIGINAL IMAGE when judging.
- Do not blindly assume the reference answer is perfect.
- Do not require exact wording.
- Do not be overly strict for minor perceptual differences.
- PARTIAL means the answer is still usable.
- AMBIGUOUS should only be used when a fair judgment cannot be made.

Question:
{question}

Ground truth:
{ground_truth}

Model answer:
{model_answer}
""".strip()


def grade_one(
    client,
    row,
    image_col,
    question_col,
    gt_col,
    answer_col,
):
    image_path = Path(
        str(row[image_col])
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    prompt = make_prompt(
        row[question_col],
        row[gt_col],
        row[answer_col],
    )

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
                        "text": prompt,
                    },
                    {
                        "type": "input_image",
                        "image_url":
                            image_to_data_url(
                                image_path
                            ),
                        "detail": "auto",
                    },
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "vqa_grade",
                "strict": True,
                "schema": GRADE_SCHEMA,
            }
        },
        max_output_tokens=1000,
    )

    result = json.loads(
        response.output_text
    )

    usage = getattr(
        response,
        "usage",
        None,
    )

    result["input_tokens"] = int(
        getattr(
            usage,
            "input_tokens",
            0,
        )
        or 0
    )

    result["output_tokens"] = int(
        getattr(
            usage,
            "output_tokens",
            0,
        )
        or 0
    )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--image-col",
        default="image_path",
    )

    parser.add_argument(
        "--question-col",
        default="question",
    )

    parser.add_argument(
        "--gt-col",
        default="ground_truth",
    )

    parser.add_argument(
        "--answer-col",
        default="model_answer",
    )

    parser.add_argument(
        "--id-col",
        default="sample_id",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS,
    )

    args = parser.parse_args()

    if not os.environ.get(
        "OPENAI_API_KEY"
    ):
        raise RuntimeError(
            "OPENAI_API_KEY is not set."
        )

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    jsonl_path = (
        output_path
        .with_suffix(".jsonl")
    )

    df = pd.read_csv(
        input_path
    )

    required = [
        args.id_col,
        args.image_col,
        args.question_col,
        args.gt_col,
        args.answer_col,
    ]

    missing = [
        x
        for x in required
        if x not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Available columns: "
            f"{list(df.columns)}"
        )

    done = set()

    if jsonl_path.exists():
        with jsonl_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                try:
                    x = json.loads(line)

                    if (
                        x.get("status")
                        == "ok"
                    ):
                        done.add(
                            str(
                                x[
                                    args.id_col
                                ]
                            )
                        )
                except Exception:
                    pass

    records = []

    for _, row in df.iterrows():
        sid = str(
            row[args.id_col]
        )

        if sid in done:
            continue

        records.append(
            row.to_dict()
        )

    print("=" * 90)
    print("IMAGE-AWARE VQA GRADING")
    print("=" * 90)
    print("Model     :", MODEL)
    print("Reasoning :", REASONING)
    print("Total     :", len(df))
    print("Completed :", len(done))
    print("Remaining :", len(records))

    client = OpenAI(
        timeout=180,
        max_retries=5,
    )

    def worker(row):
        result = grade_one(
            client,
            row,
            args.image_col,
            args.question_col,
            args.gt_col,
            args.answer_col,
        )

        return {
            args.id_col:
                str(
                    row[
                        args.id_col
                    ]
                ),

            "judge_label":
                result["label"],

            "judge_confidence":
                result["confidence"],

            "judge_reason":
                result["reason"],

            "input_tokens":
                result[
                    "input_tokens"
                ],

            "output_tokens":
                result[
                    "output_tokens"
                ],

            "status":
                "ok",

            "error":
                None,
        }

    with jsonl_path.open(
        "a",
        encoding="utf-8",
    ) as fout:

        with ThreadPoolExecutor(
            max_workers=args.workers
        ) as pool:

            futures = {
                pool.submit(
                    worker,
                    row,
                ): row
                for row in records
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Grading",
            ):
                row = futures[
                    future
                ]

                try:
                    rec = (
                        future.result()
                    )

                except Exception as e:
                    rec = {
                        args.id_col:
                            str(
                                row[
                                    args.id_col
                                ]
                            ),

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
                        rec,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                fout.flush()

    latest = {}

    with jsonl_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            try:
                x = json.loads(line)

                latest[
                    str(
                        x[
                            args.id_col
                        ]
                    )
                ] = x

            except Exception:
                pass

    grades = pd.DataFrame(
        latest.values()
    )

    df[
        args.id_col
    ] = (
        df[
            args.id_col
        ]
        .astype(str)
    )

    grades[
        args.id_col
    ] = (
        grades[
            args.id_col
        ]
        .astype(str)
    )

    out = df.merge(
        grades,
        on=args.id_col,
        how="left",
    )

    out.to_csv(
        output_path,
        index=False,
    )

    good = out[
        out["status"] == "ok"
    ].copy()

    print()
    print("=" * 90)
    print("RESULT")
    print("=" * 90)

    print(
        good[
            "judge_label"
        ]
        .value_counts()
        .to_string()
    )

    judgeable = good[
        good[
            "judge_label"
        ]
        != "AMBIGUOUS"
    ]

    usable = (
        judgeable[
            "judge_label"
        ]
        .isin(
            [
                "CORRECT",
                "PARTIAL",
            ]
        )
        .mean()
    )

    strict = (
        judgeable[
            "judge_label"
        ]
        .eq(
            "CORRECT"
        )
        .mean()
    )

    print()
    print(
        f"Usable accuracy "
        f"(CORRECT + PARTIAL): "
        f"{usable * 100:.2f}%"
    )

    print(
        f"Strict accuracy "
        f"(CORRECT only): "
        f"{strict * 100:.2f}%"
    )

    print()
    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()
