from pathlib import Path
import base64
import json
import mimetypes
import time

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


PAIRED = Path(
    "routing/benchmarks/router_training/"
    "paired_visual_router_dataset.csv"
)

AB50 = Path(
    "routing/benchmarks/superglasses/results/"
    "superglasses_large_latency_ab.csv"
)

LARGE_RESULTS = Path(
    "routing/benchmarks/superglasses/results/"
    "superglasses_large_luna_medium_1024.csv"
)

OUT_JSONL = Path(
    "routing/benchmarks/superglasses/results/"
    "superglasses_test_luna_original_latency.jsonl"
)

OUT_CSV = Path(
    "routing/benchmarks/superglasses/results/"
    "superglasses_test_luna_original_latency.csv"
)

MODEL = "gpt-5.6-luna"
REASONING = "medium"

client = OpenAI(
    timeout=120,
    max_retries=0,
)


def make_prompt(question):
    return (
        f"{question}\n\n"
        "Answer the question based on the image. "
        "Answer directly and concisely. "
        "Do not add unnecessary explanation."
    )


def original_data_url(path):
    path = Path(path)

    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "image/jpeg"

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode("ascii")

    return f"data:{mime};base64,{encoded}"


# ============================================================
# 1. Find the 111 held-out SuperGlasses test cases
# ============================================================

paired = pd.read_csv(PAIRED)

test_ids = (
    paired[
        (paired["split"] == "test")
        &
        (paired["dataset"] == "superglasses")
    ]["sample_id"]
    .astype(str)
    .tolist()
)

print("SuperGlasses test cases:", len(test_ids))

if len(test_ids) != 111:
    print(
        "WARNING: expected 111, got",
        len(test_ids)
    )


# ============================================================
# 2. Metadata / image paths
# ============================================================

meta = pd.read_csv(LARGE_RESULTS)
meta["sample_id"] = meta["sample_id"].astype(str)

meta = meta[
    meta["sample_id"].isin(test_ids)
].copy()

if len(meta) != len(test_ids):
    raise RuntimeError(
        f"Only found metadata for {len(meta)}/{len(test_ids)} test cases"
    )


# ============================================================
# 3. Reuse the existing 50 original-image measurements
# ============================================================

ab = pd.read_csv(AB50)
ab["sample_id"] = ab["sample_id"].astype(str)

existing = ab[
    (ab["mode"] == "original")
    &
    (ab["error"].isna())
    &
    (ab["sample_id"].isin(test_ids))
].copy()

existing_records = {}

for _, row in existing.iterrows():
    existing_records[str(row["sample_id"])] = {
        "sample_id": str(row["sample_id"]),
        "latency_sec": float(row["latency_sec"]),
        "input_tokens": row.get("input_tokens"),
        "output_tokens": row.get("output_tokens"),
        "answer": row.get("answer"),
        "source": "existing_ab50",
        "status": "ok",
        "error": None,
    }

print("Existing original measurements:", len(existing_records))


# ============================================================
# 4. Resume any measurements from this script
# ============================================================

new_records = {}

if OUT_JSONL.exists():
    with OUT_JSONL.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            if not line.strip():
                continue

            try:
                x = json.loads(line)

                if (
                    x.get("status") == "ok"
                    and x.get("sample_id")
                ):
                    new_records[
                        str(x["sample_id"])
                    ] = x

            except Exception:
                pass


completed_ids = (
    set(existing_records)
    |
    set(new_records)
)

remaining = meta[
    ~meta["sample_id"].isin(completed_ids)
].copy()

print("Already measured total :", len(completed_ids))
print("Remaining              :", len(remaining))
print()


# ============================================================
# 5. Run ONLY missing original-image Luna calls
# ============================================================

OUT_JSONL.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUT_JSONL.open(
    "a",
    encoding="utf-8",
) as fout:

    for _, row in tqdm(
        remaining.iterrows(),
        total=len(remaining),
        desc="Luna original test latency",
    ):

        sid = str(row["sample_id"])
        image_path = Path(row["image_path"])

        try:
            image_url = original_data_url(
                image_path
            )

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
                                    str(row["question"])
                                ),
                            },
                            {
                                "type": "input_image",
                                "image_url": image_url,
                                "detail": "auto",
                            },
                        ],
                    }
                ],

                # Match the previous 50-case A/B test.
                max_output_tokens=500,
                store=False,
            )

            latency = (
                time.perf_counter()
                - t0
            )

            answer = (
                response.output_text
                or ""
            ).strip()

            if not answer:
                raise RuntimeError(
                    "Luna returned empty answer"
                )

            usage = getattr(
                response,
                "usage",
                None,
            )

            record = {
                "sample_id": sid,
                "question": row["question"],
                "image_path": str(image_path),
                "latency_sec": latency,
                "input_tokens": getattr(
                    usage,
                    "input_tokens",
                    None,
                ),
                "output_tokens": getattr(
                    usage,
                    "output_tokens",
                    None,
                ),
                "answer": answer,
                "source": "new_original_test",
                "status": "ok",
                "error": None,
            }

        except Exception as e:

            record = {
                "sample_id": sid,
                "question": row["question"],
                "image_path": str(image_path),
                "latency_sec": None,
                "input_tokens": None,
                "output_tokens": None,
                "answer": None,
                "source": "new_original_test",
                "status": "error",
                "error": (
                    f"{type(e).__name__}: {e}"
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
# 6. Reload + combine old 50 and new measurements
# ============================================================

new_records = {}

if OUT_JSONL.exists():
    with OUT_JSONL.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            try:
                x = json.loads(line)
                new_records[
                    str(x["sample_id"])
                ] = x
            except Exception:
                pass


combined = {}

combined.update(
    existing_records
)

for sid, record in new_records.items():

    if (
        record.get("status")
        == "ok"
    ):
        combined[sid] = record


rows = []

for sid in test_ids:

    record = combined.get(
        sid,
        {
            "sample_id": sid,
            "status": "missing",
        },
    )

    rows.append(record)


out = pd.DataFrame(rows)

out.to_csv(
    OUT_CSV,
    index=False,
)


# ============================================================
# 7. Final confirmed test latency
# ============================================================

good = out[
    out["status"] == "ok"
].copy()

lat = pd.to_numeric(
    good["latency_sec"],
    errors="coerce",
).dropna()


print()
print("=" * 80)
print("SUPERGLASSES TEST — LUNA ORIGINAL")
print("=" * 80)

print(
    "Test cases       :",
    len(test_ids),
)

print(
    "Measured         :",
    len(lat),
)

print(
    "Missing/errors   :",
    len(test_ids) - len(lat),
)

if len(lat):

    print(
        f"Mean latency     : {lat.mean():.3f}s"
    )

    print(
        f"Median latency   : {lat.median():.3f}s"
    )

    print(
        f"P95 latency      : {lat.quantile(.95):.3f}s"
    )

    print(
        f"Min latency      : {lat.min():.3f}s"
    )

    print(
        f"Max latency      : {lat.max():.3f}s"
    )


print()
print("Saved:")
print(OUT_CSV)
