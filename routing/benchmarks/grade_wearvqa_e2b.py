from pathlib import Path
import json
import os
import time

import pandas as pd
from tqdm import tqdm
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "routing/benchmarks/results/"
    / "wearvqa_e2b_full_raw.csv"
)

OUT_JSONL = (
    ROOT
    / "routing/benchmarks/results/"
    / "wearvqa_e2b_full_graded.jsonl"
)

OUT_CSV = (
    ROOT
    / "routing/benchmarks/results/"
    / "wearvqa_e2b_full_graded.csv"
)

MODEL = os.environ.get(
    "WEARVQA_JUDGE_MODEL",
    "gpt-5.6-luna",
)

client = OpenAI()


def make_prompt(question, gt, pred):
    return f"""
You are grading the answer of a vision-language model.

Determine whether the MODEL ANSWER correctly answers the QUESTION,
using the REFERENCE ANSWER as the intended ground truth.

Important grading rules:
- Do NOT require exact wording.
- Paraphrases are correct.
- Equivalent numbers/units are correct.
- Extra harmless detail is allowed.
- If the model gives the wrong object, number, text, relation,
  activity, purpose, or conclusion, mark INCORRECT.
- The reference answer may be more verbose than necessary.
- If the model answer could reasonably be correct but the reference
  alone is insufficient to decide, mark AMBIGUOUS.
- Do not reward vague answers that fail to provide the requested fact.

QUESTION:
{question}

REFERENCE ANSWER:
{gt}

MODEL ANSWER:
{pred}

Return exactly one line of JSON in this format:
{{"label":"CORRECT","reason":"short reason"}}

label must be exactly one of:
CORRECT
INCORRECT
AMBIGUOUS
""".strip()


# ------------------------------------------------------------
# Resume
# ------------------------------------------------------------

completed = {}

if OUT_JSONL.exists():
    with OUT_JSONL.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            try:
                x = json.loads(line)
                completed[str(x["sample_id"])] = x
            except Exception:
                pass


df = pd.read_csv(INPUT)

print("=" * 80)
print("WEARVQA E2B SEMANTIC GRADING")
print("=" * 80)
print("Model             :", MODEL)
print("Total             :", len(df))
print("Already completed :", len(completed))
print("Remaining         :", len(df) - len(completed))
print()


with OUT_JSONL.open(
    "a",
    encoding="utf-8",
) as fout:

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Judging WearVQA",
    ):

        sid = str(row["sample_id"])

        if sid in completed:
            continue

        record = row.to_dict()

        question = str(row["question"])
        gt = str(row["ground_truth"])
        pred = str(row["prediction"])

        try:
            response = client.responses.create(
                model=MODEL,
                input=make_prompt(
                    question,
                    gt,
                    pred,
                ),
            )

            text = response.output_text.strip()

            # tolerate accidental markdown fences
            text = (
                text
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

            parsed = json.loads(text)

            label = str(
                parsed["label"]
            ).upper()

            if label not in {
                "CORRECT",
                "INCORRECT",
                "AMBIGUOUS",
            }:
                raise ValueError(
                    f"Unexpected label: {label}"
                )

            record["judge_label"] = label
            record["judge_reason"] = str(
                parsed.get("reason", "")
            )

            record["judge_error"] = None

        except Exception as e:
            record["judge_label"] = None
            record["judge_reason"] = None
            record["judge_error"] = (
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

        completed[sid] = record

        # gentle rate limiting
        time.sleep(0.05)


# ------------------------------------------------------------
# Rebuild CSV
# ------------------------------------------------------------

rows = []

with OUT_JSONL.open(
    "r",
    encoding="utf-8",
) as f:
    for line in f:
        try:
            rows.append(
                json.loads(line)
            )
        except Exception:
            pass


out = pd.DataFrame(rows)

out = (
    out
    .drop_duplicates(
        subset=["sample_id"],
        keep="last",
    )
    .reset_index(drop=True)
)

out.to_csv(
    OUT_CSV,
    index=False,
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print()
print("=" * 80)
print("WEARVQA E2B GRADING SUMMARY")
print("=" * 80)

print("Cases:", len(out))

print()
print("Labels:")
print(
    out["judge_label"]
    .value_counts(
        dropna=False
    )
    .to_string()
)

valid = out[
    out["judge_label"].isin(
        ["CORRECT", "INCORRECT"]
    )
].copy()

if len(valid):
    acc = (
        valid["judge_label"]
        == "CORRECT"
    ).mean()

    print()
    print(
        f"E2B accuracy "
        f"(excluding ambiguous): "
        f"{acc*100:.2f}%"
    )

    print()
    print("Accuracy by category:")

    valid["correct"] = (
        valid["judge_label"]
        == "CORRECT"
    ).astype(int)

    table = (
        valid
        .groupby("category")
        ["correct"]
        .agg(["count", "mean"])
        .sort_values("mean")
    )

    table["mean"] *= 100

    print(table.to_string())


print()
print("Saved:")
print(OUT_CSV)
