from pathlib import Path
import base64
import io
import mimetypes
import os
import time

import pandas as pd
from PIL import Image, ImageOps
from openai import OpenAI
from tqdm import tqdm


ROOT = Path("routing/benchmarks/superglasses")

INPUT = (
    ROOT
    / "results"
    / "superglasses_large_luna_medium.csv"
)

OUT = (
    ROOT
    / "results"
    / "superglasses_large_latency_ab.csv"
)

MODEL = "gpt-5.6-luna"
REASONING = "medium"

N = 50

client = OpenAI(
    timeout=120,
    max_retries=0,   # important: don't hide retry time
)


def original_data_url(path):
    path = Path(path)

    mime, _ = mimetypes.guess_type(
        str(path)
    )

    if not mime:
        mime = "image/jpeg"

    data = base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")

    return f"data:{mime};base64,{data}"


def resized_data_url(path, max_edge=1024):
    path = Path(path)

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")

        im.thumbnail(
            (max_edge, max_edge),
            Image.Resampling.LANCZOS,
        )

        buf = io.BytesIO()

        im.save(
            buf,
            format="JPEG",
            quality=90,
        )

    data = base64.b64encode(
        buf.getvalue()
    ).decode("utf-8")

    return f"data:image/jpeg;base64,{data}"


def make_prompt(question):
    return (
        f"{question}\n\n"
        "Answer the question based on the image. "
        "Answer directly and concisely. "
        "Do not add unnecessary explanation."
    )


def run_one(row, mode):
    if mode == "original":
        image_url = original_data_url(
            row["image_path"]
        )
    else:
        image_url = resized_data_url(
            row["image_path"],
            1024,
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
                            row["question"]
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
        max_output_tokens=500,
    )

    latency = (
        time.perf_counter()
        - t0
    )

    usage = response.usage

    return {
        "sample_id": row["sample_id"],
        "mode": mode,
        "latency_sec": latency,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "answer": response.output_text.strip(),
    }


df = pd.read_csv(INPUT)

# Same deterministic 50 cases for both tests
df = (
    df[df["status"] == "ok"]
    .sort_values("sample_id")
    .head(N)
)

records = []

for mode in [
    "original",
    "resize_1024",
]:
    print()
    print("=" * 80)
    print(mode)
    print("=" * 80)

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
    ):
        try:
            result = run_one(
                row,
                mode,
            )
            result["error"] = None

        except Exception as e:
            result = {
                "sample_id":
                    row["sample_id"],
                "mode":
                    mode,
                "latency_sec":
                    None,
                "input_tokens":
                    None,
                "output_tokens":
                    None,
                "answer":
                    None,
                "error":
                    f"{type(e).__name__}: {e}",
            }

        records.append(result)

out = pd.DataFrame(records)

out.to_csv(
    OUT,
    index=False,
)

print()
print("=" * 80)
print("RESULT")
print("=" * 80)

for mode in [
    "original",
    "resize_1024",
]:
    x = out[
        (out["mode"] == mode)
        & out["error"].isna()
    ]

    print()
    print(mode)
    print(
        "Successful:",
        len(x),
    )
    print(
        "Mean latency:",
        round(
            x["latency_sec"].mean(),
            3,
        ),
    )
    print(
        "Median latency:",
        round(
            x["latency_sec"].median(),
            3,
        ),
    )
    print(
        "P95 latency:",
        round(
            x["latency_sec"].quantile(0.95),
            3,
        ),
    )
    print(
        "Mean input tokens:",
        round(
            x["input_tokens"].mean(),
            1,
        ),
    )

print()
print("Saved:")
print(OUT)
