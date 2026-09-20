from pathlib import Path
import base64
import json
import os
import mimetypes

import pandas as pd
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "routing/benchmarks/results/"
    / "wearvqa_e2b_50_graded.csv"
)

OUTPUT = (
    ROOT
    / "routing/benchmarks/results/"
    / "wearvqa_image_judge_50.csv"
)

MODEL = os.environ.get(
    "WEARVQA_JUDGE_MODEL",
    "gpt-5.6-luna",
)

client = OpenAI()


def image_to_data_url(path: Path):
    mime, _ = mimetypes.guess_type(str(path))

    if mime is None:
        mime = "image/jpeg"

    data = base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")

    return f"data:{mime};base64,{data}"


def make_prompt(question, gt, pred):
    return f"""
You are evaluating a SMALL vision-language model used in wearable AI glasses.

You can see the actual image.

Your goal is NOT to require an exact match to the reference answer.
Your goal is to determine whether the SMALL model's answer is useful
and sufficiently correct for the person wearing the glasses.

Use BOTH:
1. the actual image
2. the reference answer

The reference answer is guidance, but it may be overly specific,
imperfect, or describe a visually ambiguous attribute.

Labels:

CORRECT
- The core information requested by the user is correct.
- Minor omissions or harmless wording differences are okay.

PARTIAL
- The answer is substantially useful and close enough for the user,
  but has a minor perceptual difference, missing detail, or imprecision.
- Examples include visually similar colors, nearby fine-grained object
  categories, approximate positions, or omission of non-essential details.
- PARTIAL should still mean that using a LARGE model may not be necessary.

INCORRECT
- The core requested information is clearly wrong.
- The answer contradicts what is clearly visible.
- It gives a clearly wrong count, object, action, relation, text, or conclusion.
- It answers the wrong question or fails to provide the requested information.

AMBIGUOUS
- The image itself does not allow a confident decision.
- Both the reference and model answer are visually plausible.
- The reference may be questionable.
- The distinction is too subtle to judge reliably.

IMPORTANT:
Do NOT mark INCORRECT merely because the prediction differs from the
reference wording.

If two nearby colors such as red/pink or green/yellow are genuinely
hard to distinguish in the image, prefer PARTIAL or AMBIGUOUS.

If two fine-grained categories are visually difficult to distinguish,
prefer PARTIAL when the model still gives useful information.

However, do NOT use PARTIAL to excuse a clearly wrong core answer.

QUESTION:
{question}

REFERENCE ANSWER:
{gt}

SMALL MODEL ANSWER:
{pred}

Return exactly one JSON object:

{{"label":"PARTIAL","reason":"short reason"}}

label must be exactly one of:
CORRECT
PARTIAL
INCORRECT
AMBIGUOUS
""".strip()


df = pd.read_csv(INPUT)

FULL_MANIFEST = (
    ROOT
    / "routing/benchmarks/wearvqa_full/manifest.csv"
)

full_manifest = pd.read_csv(FULL_MANIFEST)

rows = []

print("=" * 90)
print("IMAGE-AWARE WEARABLE JUDGE — 50 CASE VALIDATION")
print("=" * 90)
print("Model:", MODEL)
print("Cases:", len(df))
print()


for i, r in df.iterrows():

    # The historical 50-case mini set may use different sample IDs.
    # Match the same WearVQA example using its question text, then
    # read the actual image path recorded in the full manifest.
    question_text = str(r["question"]).strip()

    matches = full_manifest[
        full_manifest["question"]
        .astype(str)
        .str.strip()
        .eq(question_text)
    ]

    # If question text alone is not unique, also use ground truth.
    if len(matches) > 1:
        gt_text = str(r["ground_truth"]).strip()

        matches_gt = matches[
            matches["ground_truth"]
            .astype(str)
            .str.strip()
            .eq(gt_text)
        ]

        if len(matches_gt):
            matches = matches_gt

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not match mini case to full WearVQA: "
            f"sample_id={r['sample_id']}, "
            f"question={question_text}"
        )

    full_row = matches.iloc[0]

    image_path = Path(
        str(full_row["image"])
    )

    if not image_path.is_absolute():
        image_path = ROOT / image_path

    if not image_path.exists():
        raise FileNotFoundError(
            f"Matched WearVQA case, but image is missing: "
            f"{image_path}"
        )

    manual = str(
        r["manual_correct"]
    ).strip().lower()

    if manual in {"1", "1.0"}:
        manual_label = "CORRECT"

    elif manual in {"0", "0.0"}:
        manual_label = "INCORRECT"

    else:
        manual_label = "AMBIGUOUS"

    try:
        response = client.responses.create(
            model=MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": make_prompt(
                                str(r["question"]),
                                str(r["ground_truth"]),
                                str(r["prediction"]),
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": image_to_data_url(
                                image_path
                            ),
                        },
                    ],
                }
            ],
        )

        text = (
            response.output_text
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        parsed = json.loads(text)

        label = str(
            parsed["label"]
        ).upper()

        reason = str(
            parsed.get("reason", "")
        )

        if label not in {
            "CORRECT",
            "PARTIAL",
            "INCORRECT",
            "AMBIGUOUS",
        }:
            raise ValueError(
                f"Unexpected label: {label}"
            )

        error = None

    except Exception as e:
        label = None
        reason = None
        error = f"{type(e).__name__}: {e}"

    rows.append({
        "sample_id": r["sample_id"],
        "category": r["category"],
        "image": r["image"],
        "question": r["question"],
        "ground_truth": r["ground_truth"],
        "prediction": r["prediction"],
        "manual_label": manual_label,
        "judge_label": label,
        "judge_reason": reason,
        "error": error,
    })

    print(
        f"{i+1:02d}/50  "
        f"human={manual_label:<9} "
        f"judge={str(label):<9}"
    )


out = pd.DataFrame(rows)

out.to_csv(
    OUTPUT,
    index=False,
)


print()
print("=" * 90)
print("LABEL DISTRIBUTION")
print("=" * 90)

print(
    out["judge_label"]
    .value_counts(dropna=False)
    .to_string()
)


# ============================================================
# Compare USER-UTILITY binary labels
#
# Human:
#   CORRECT   -> ACCEPTABLE
#   INCORRECT -> FAILURE
#
# Image judge:
#   CORRECT + PARTIAL -> ACCEPTABLE
#   INCORRECT         -> FAILURE
#
# Ambiguous excluded.
# ============================================================

def human_binary(x):
    if x == "CORRECT":
        return "ACCEPTABLE"
    if x == "INCORRECT":
        return "FAILURE"
    return None


def judge_binary(x):
    if x in {"CORRECT", "PARTIAL"}:
        return "ACCEPTABLE"
    if x == "INCORRECT":
        return "FAILURE"
    return None


out["human_binary"] = (
    out["manual_label"]
    .map(human_binary)
)

out["judge_binary"] = (
    out["judge_label"]
    .map(judge_binary)
)

valid = out[
    out["human_binary"].notna()
    & out["judge_binary"].notna()
].copy()

agreement = (
    valid["human_binary"]
    == valid["judge_binary"]
).mean()

print()
print("=" * 90)
print("USER-UTILITY AGREEMENT")
print("=" * 90)

print(
    f"Judgeable cases : {len(valid)}"
)

print(
    f"Agreement       : "
    f"{agreement*100:.2f}%"
)


print()
print("=" * 90)
print("DISAGREEMENTS")
print("=" * 90)

bad = valid[
    valid["human_binary"]
    != valid["judge_binary"]
]

print("Count:", len(bad))

for _, r in bad.iterrows():

    print()
    print("-" * 90)
    print("ID    :", r["sample_id"])
    print("CAT   :", r["category"])
    print("Q     :", r["question"])
    print("GT    :", r["ground_truth"])
    print("E2B   :", r["prediction"])
    print("HUMAN :", r["manual_label"])
    print("JUDGE :", r["judge_label"])
    print("WHY   :", r["judge_reason"])


print()
print("=" * 90)
print("PARTIAL CASES")
print("=" * 90)

partial = out[
    out["judge_label"] == "PARTIAL"
]

print("Count:", len(partial))

for _, r in partial.iterrows():

    print()
    print("-" * 90)
    print("ID  :", r["sample_id"])
    print("Q   :", r["question"])
    print("GT  :", r["ground_truth"])
    print("E2B :", r["prediction"])
    print("WHY :", r["judge_reason"])


out.to_csv(
    OUTPUT,
    index=False,
)

print()
print("Saved:")
print(OUTPUT)
