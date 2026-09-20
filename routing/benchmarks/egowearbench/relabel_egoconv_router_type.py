from pathlib import Path
import json
import os

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


INPUT = Path(
    "routing/benchmarks/egowearbench/"
    "question_manifest.csv"
)

OUT_JSONL = Path(
    "routing/benchmarks/egowearbench/"
    "egoconv_routing_labels.jsonl"
)

OUT_CSV = Path(
    "routing/benchmarks/egowearbench/"
    "egoconv_routing_labels.csv"
)

MODEL = os.environ.get(
    "ROUTER_LABEL_MODEL",
    "gpt-5.6-luna",
)

client = OpenAI()


def prompt(question):
    return f"""
You are labeling a question for an AI glasses routing system.

Classify what information is fundamentally required to answer the question.

Use exactly one label:

DIRECT_VISUAL
- Can normally be answered from the current visual scene or a single relevant frame.
- Includes object recognition, attributes, OCR, counting, spatial relations,
  current scene understanding, and questions such as "what is this?",
  "what color is it?", "what does the sign say?", or "where is the cup?"

TEMPORAL
- Requires observing change, actions, order, or events across multiple moments.
- Includes before/after, what happened, how an action unfolded,
  what someone was doing over time, sequence, motion, or next-event questions.
- Do NOT label TEMPORAL merely because the data came from a video.
  If one frame is enough, use DIRECT_VISUAL.

MEMORY
- Requires recalling information from earlier personal experience beyond
  the immediately relevant current video segment.
- Includes "where did I leave...", "what did someone tell me earlier?",
  "what did I say before?", or long-term episodic/personal memory.

KNOWLEDGE
- Requires factual or external information not visibly contained in the scene.
- Includes identifying a visible entity and then needing outside facts,
  history, product information, dates, materials, company facts, or web search.
- If the answer can be directly read or recognized from the image, do NOT use KNOWLEDGE.

Important:
Classify based on what the QUESTION fundamentally requires, not based on
which dataset it came from.

Question:
{question}

Return exactly one JSON object:
{{"label":"DIRECT_VISUAL","reason":"short reason"}}
""".strip()


df = pd.read_csv(INPUT)

df = df[
    df["task"] == "egoconv"
].copy()

completed = {}

if OUT_JSONL.exists():
    with OUT_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                x = json.loads(line)
                completed[str(x["sample_id"])] = x
            except Exception:
                pass


print("=" * 90)
print("RELABEL EGOCONV ROUTING TYPE")
print("=" * 90)

print("Model             :", MODEL)
print("Total             :", len(df))
print("Already completed :", len(completed))
print("Remaining         :", len(df) - len(completed))
print()


with OUT_JSONL.open("a", encoding="utf-8") as fout:

    for _, r in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Relabel EgoConv",
    ):

        sid = str(r["sample_id"])

        if sid in completed:
            continue

        record = {
            "sample_id": sid,
            "question": r["question"],
            "old_routing_type": r["routing_type"],
        }

        try:
            response = client.responses.create(
                model=MODEL,
                input=prompt(str(r["question"])),
            )

            text = (
                response.output_text
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

            parsed = json.loads(text)

            label = parsed["label"].strip().upper()

            if label not in {
                "DIRECT_VISUAL",
                "TEMPORAL",
                "MEMORY",
                "KNOWLEDGE",
            }:
                raise ValueError(
                    f"Unexpected label: {label}"
                )

            record["new_routing_type"] = label
            record["reason"] = parsed.get(
                "reason",
                "",
            )
            record["error"] = None

        except Exception as e:
            record["new_routing_type"] = None
            record["reason"] = None
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

        completed[sid] = record


rows = []

with OUT_JSONL.open("r", encoding="utf-8") as f:
    for line in f:
        try:
            rows.append(json.loads(line))
        except Exception:
            pass

out = pd.DataFrame(rows)

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


print()
print("=" * 90)
print("NEW ROUTING DISTRIBUTION")
print("=" * 90)

print(
    out["new_routing_type"]
    .value_counts(dropna=False)
    .to_string()
)


print()
print("=" * 90)
print("EXAMPLES BY NEW LABEL")
print("=" * 90)

for label in [
    "DIRECT_VISUAL",
    "TEMPORAL",
    "MEMORY",
    "KNOWLEDGE",
]:

    x = out[
        out["new_routing_type"] == label
    ]

    print()
    print(f"[{label}] n={len(x)}")

    for _, r in x.head(10).iterrows():
        print(
            "-",
            r["question"],
            "|",
            r["reason"],
        )


print()
print("Saved:")
print(OUT_CSV)
