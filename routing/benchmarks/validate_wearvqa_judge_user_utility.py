import json
import os

import pandas as pd
from openai import OpenAI
from sklearn.metrics import classification_report, confusion_matrix


INPUT = (
    "routing/benchmarks/results/"
    "wearvqa_e2b_50_graded.csv"
)

OUTPUT = (
    "routing/benchmarks/results/"
    "wearvqa_judge_user_utility_50.csv"
)

MODEL = os.environ.get(
    "WEARVQA_JUDGE_MODEL",
    "gpt-5.6-luna",
)

client = OpenAI()


def make_prompt(question, gt, pred):
    return f"""
You are evaluating the answer of a SMALL vision-language model
used in wearable AI glasses.

Your goal is NOT to check whether the model reproduces every detail
of the reference answer.

Instead, determine whether the MODEL ANSWER is sufficiently correct
and useful for the user who asked the QUESTION.

Use the REFERENCE ANSWER as guidance for the intended answer.

Grading principles:

1. Mark CORRECT if the model gives the core information requested
   by the question correctly.

2. A shorter or less specific answer can still be CORRECT if it
   answers the main user intent.

3. Do NOT mark an answer incorrect only because it omits secondary,
   non-essential details from the reference.

4. Paraphrases and semantically equivalent expressions are CORRECT.

5. Equivalent numbers, units, object names, or common synonyms
   should be accepted when they preserve the intended meaning.

6. Mark INCORRECT if:
   - the core requested information is wrong,
   - the answer contradicts the reference,
   - an important requested detail is missing,
   - the answer addresses the wrong aspect of the question,
   - or the answer is too vague to satisfy the question.

7. If the reference answer alone is insufficient to confidently
   determine whether the model answer is acceptable, mark AMBIGUOUS.

Examples:

Question:
What is the leash attached to?

Reference:
The leash is attached to a dog harness on a dog.

Model:
The leash is attached to a dog.

Label:
CORRECT

Reason:
The model gives the main requested information, although it omits
the more specific harness detail.


Question:
How many water bottles are on the counter?

Reference:
There are two water bottles on the counter.

Model:
There are no water bottles.

Label:
INCORRECT

Reason:
The core requested count is wrong.


Question:
What is the purpose of the furry object around their arms?

Reference:
It is used as a fashion accessory and for warmth.

Model:
It appears to be a fur boa.

Label:
INCORRECT

Reason:
The model identifies the object but does not answer the requested purpose.


Now grade this case:

QUESTION:
{question}

REFERENCE ANSWER:
{gt}

MODEL ANSWER:
{pred}

Return exactly one line of JSON:

{{"label":"CORRECT","reason":"short reason"}}

The label must be exactly one of:
CORRECT
INCORRECT
AMBIGUOUS
""".strip()


df = pd.read_csv(INPUT)

rows = []

print("=" * 90)
print("VALIDATE USER-UTILITY GPT JUDGE AGAINST HUMAN LABELS")
print("=" * 90)
print("Model:", MODEL)
print("Cases:", len(df))
print()


for i, r in df.iterrows():

    manual = str(r["manual_correct"]).strip().lower()

    if manual in {"1", "1.0"}:
        manual_label = "CORRECT"

    elif manual in {"0", "0.0"}:
        manual_label = "INCORRECT"

    else:
        manual_label = "AMBIGUOUS"

    try:
        response = client.responses.create(
            model=MODEL,
            input=make_prompt(
                str(r["question"]),
                str(r["ground_truth"]),
                str(r["prediction"]),
            ),
        )

        text = (
            response.output_text
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        parsed = json.loads(text)

        gpt_label = str(
            parsed["label"]
        ).upper()

        gpt_reason = str(
            parsed.get("reason", "")
        )

        if gpt_label not in {
            "CORRECT",
            "INCORRECT",
            "AMBIGUOUS",
        }:
            raise ValueError(
                f"Unexpected label: {gpt_label}"
            )

        error = None

    except Exception as e:
        gpt_label = None
        gpt_reason = None
        error = f"{type(e).__name__}: {e}"

    rows.append({
        "sample_id": r["sample_id"],
        "category": r["category"],
        "question": r["question"],
        "ground_truth": r["ground_truth"],
        "prediction": r["prediction"],
        "manual_label": manual_label,
        "gpt_label": gpt_label,
        "gpt_reason": gpt_reason,
        "error": error,
    })

    print(
        f"{i+1:02d}/{len(df)}  "
        f"human={manual_label:<9}  "
        f"GPT={str(gpt_label):<9}"
    )


out = pd.DataFrame(rows)

out.to_csv(
    OUTPUT,
    index=False,
)


print()
print("=" * 90)
print("SUMMARY")
print("=" * 90)

print()
print("Human labels:")
print(
    out["manual_label"]
    .value_counts(dropna=False)
    .to_string()
)

print()
print("GPT labels:")
print(
    out["gpt_label"]
    .value_counts(dropna=False)
    .to_string()
)


# ------------------------------------------------------------
# Agreement on human judgeable cases
# ------------------------------------------------------------

valid = out[
    out["manual_label"].isin(
        ["CORRECT", "INCORRECT"]
    )
].copy()

agreement = (
    valid["manual_label"]
    == valid["gpt_label"]
).mean()

print()
print(
    f"Overall agreement on human-judgeable cases: "
    f"{agreement*100:.2f}%"
)


# ------------------------------------------------------------
# Binary classification report
# ------------------------------------------------------------

binary = valid[
    valid["gpt_label"].isin(
        ["CORRECT", "INCORRECT"]
    )
].copy()

if len(binary):

    print()
    print("Confusion matrix")
    print(
        "(rows = HUMAN, columns = GPT)"
    )
    print(
        confusion_matrix(
            binary["manual_label"],
            binary["gpt_label"],
            labels=[
                "CORRECT",
                "INCORRECT",
            ],
        )
    )

    print()
    print(
        classification_report(
            binary["manual_label"],
            binary["gpt_label"],
            labels=[
                "CORRECT",
                "INCORRECT",
            ],
            zero_division=0,
        )
    )


# ------------------------------------------------------------
# Disagreements
# ------------------------------------------------------------

bad = out[
    out["manual_label"]
    != out["gpt_label"]
]

print()
print("=" * 90)
print(
    f"DISAGREEMENTS: {len(bad)}"
)
print("=" * 90)

for _, r in bad.iterrows():

    print()
    print("-" * 90)

    print(
        "ID    :",
        r["sample_id"],
    )

    print(
        "CAT   :",
        r["category"],
    )

    print(
        "Q     :",
        r["question"],
    )

    print(
        "GT    :",
        r["ground_truth"],
    )

    print(
        "E2B   :",
        r["prediction"],
    )

    print(
        "HUMAN :",
        r["manual_label"],
    )

    print(
        "GPT   :",
        r["gpt_label"],
    )

    print(
        "WHY   :",
        r["gpt_reason"],
    )


print()
print("=" * 90)
print("SAVED")
print("=" * 90)
print(OUTPUT)
