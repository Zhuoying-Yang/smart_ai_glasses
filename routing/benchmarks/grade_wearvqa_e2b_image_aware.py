from pathlib import Path
import base64
import json
import mimetypes
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

MANIFEST = (
    ROOT
    / "routing/benchmarks/wearvqa_full/"
    / "manifest.csv"
)

OUT_JSONL = (
    ROOT
    / "routing/benchmarks/results/"
    / "wearvqa_e2b_image_graded.jsonl"
)

OUT_CSV = (
    ROOT
    / "routing/benchmarks/results/"
    / "wearvqa_e2b_image_graded.csv"
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
Determine whether the SMALL model's answer is sufficiently correct
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
- PARTIAL means a LARGE model may not be necessary.

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
manifest = pd.read_csv(MANIFEST)

# sample_id -> true image path
image_map = {}

for _, r in manifest.iterrows():
    image_map[str(r["sample_id"])] = str(r["image"])


# ============================================================
# Resume
# ============================================================

completed = {}

if OUT_JSONL.exists():
    with OUT_JSONL.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                x = json.loads(line)
                completed[str(x["sample_id"])] = x
            except Exception:
                pass


print("=" * 90)
print("WEARVQA FULL IMAGE-AWARE GRADING")
print("=" * 90)

print("Model             :", MODEL)
print("Total             :", len(df))
print("Already completed :", len(completed))
print("Remaining         :", len(df) - len(completed))
print()


with OUT_JSONL.open(
    "a",
    encoding="utf-8",
) as fout:

    for _, r in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Image-aware judging",
    ):

        sid = str(r["sample_id"])

        if sid in completed:
            continue

        record = r.to_dict()

        try:
            if sid not in image_map:
                raise KeyError(
                    f"sample_id {sid} missing from manifest"
                )

            image_path = Path(
                image_map[sid]
            )

            if not image_path.is_absolute():
                image_path = ROOT / image_path

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Missing image: {image_path}"
                )

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

            if label not in {
                "CORRECT",
                "PARTIAL",
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

        time.sleep(0.03)


# ============================================================
# Rebuild CSV
# ============================================================

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

# Router binary target
def router_target(label):
    if label in {
        "CORRECT",
        "PARTIAL",
    }:
        return 0

    if label == "INCORRECT":
        return 1

    return None


out["small_failure"] = (
    out["judge_label"]
    .map(router_target)
)

out.to_csv(
    OUT_CSV,
    index=False,
)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 90)
print("IMAGE-AWARE GRADING SUMMARY")
print("=" * 90)

print("Cases:", len(out))

print()
print("Labels:")
print(
    out["judge_label"]
    .value_counts(dropna=False)
    .to_string()
)

judgeable = out[
    out["small_failure"].notna()
].copy()

if len(judgeable):

    failure_rate = (
        judgeable["small_failure"]
        .astype(float)
        .mean()
    )

    print()
    print(
        f"SMALL acceptable rate : "
        f"{(1-failure_rate)*100:.2f}%"
    )

    print(
        f"SMALL failure rate    : "
        f"{failure_rate*100:.2f}%"
    )

    print(
        f"Judgeable cases       : "
        f"{len(judgeable):,}"
    )


    print()
    print("Failure rate by category:")

    table = (
        judgeable
        .groupby("category")
        ["small_failure"]
        .agg(["count", "mean"])
        .sort_values("mean")
    )

    table["mean"] *= 100

    print(
        table.to_string()
    )


print()
print("Saved:")
print(OUT_CSV)
