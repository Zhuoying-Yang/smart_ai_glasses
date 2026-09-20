from pathlib import Path
import base64
import io
import shutil
import time

import pandas as pd
from PIL import Image, ImageOps
from openai import OpenAI
from tqdm import tqdm


INPUT = Path(
    "routing/benchmarks/superglasses/results/"
    "superglasses_large_luna_medium_1024.csv"
)

BACKUP = Path(
    "routing/benchmarks/superglasses/results/"
    "superglasses_large_luna_medium_1024_before_retry.csv"
)

MODEL = "gpt-5.6-luna"
REASONING = "medium"
MAX_EDGE = 1024

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


def resize_image(path):
    path = Path(path)

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")

        im.thumbnail(
            (MAX_EDGE, MAX_EDGE),
            Image.Resampling.LANCZOS,
        )

        buf = io.BytesIO()

        im.save(
            buf,
            format="JPEG",
            quality=90,
        )

    encoded = base64.b64encode(
        buf.getvalue()
    ).decode("ascii")

    return f"data:image/jpeg;base64,{encoded}"


df = pd.read_csv(INPUT)

# Make backup once.
if not BACKUP.exists():
    shutil.copy2(INPUT, BACKUP)
    print("Backup:", BACKUP)


# Find rows with no LARGE answer.
answer_col = "large_answer"

if answer_col not in df.columns:
    raise RuntimeError(
        f"{answer_col} column not found."
    )

missing_mask = (
    df[answer_col].isna()
    |
    (df[answer_col].astype(str).str.strip() == "")
)

missing = df[missing_mask].copy()

print("=" * 90)
print("RETRY MISSING LARGE ANSWERS")
print("=" * 90)
print("Missing:", len(missing))
print()


for idx, row in tqdm(
    missing.iterrows(),
    total=len(missing),
    desc="Luna-medium retry",
):

    image_path = Path(
        str(row["image_path"])
    )

    if not image_path.exists():
        print(
            f"\n{row['sample_id']}: "
            f"IMAGE MISSING: {image_path}"
        )
        continue

    try:
        image_url = resize_image(
            image_path
        )

        started = time.perf_counter()

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
                                    str(
                                        row[
                                            "question"
                                        ]
                                    )
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

            # Some of these are long OCR /
            # translation questions, so give
            # more room than the original run.
            max_output_tokens=4096,

            store=False,
        )

        latency = (
            time.perf_counter()
            - started
        )

        answer = (
            response.output_text
            or ""
        ).strip()

        if not answer:
            raise RuntimeError(
                "Luna returned empty answer"
            )

        usage = response.usage

        df.at[
            idx,
            "large_answer"
        ] = answer

        df.at[
            idx,
            "latency_sec"
        ] = latency

        if (
            "input_tokens"
            in df.columns
        ):
            df.at[
                idx,
                "input_tokens"
            ] = getattr(
                usage,
                "input_tokens",
                None,
            )

        if (
            "output_tokens"
            in df.columns
        ):
            df.at[
                idx,
                "output_tokens"
            ] = getattr(
                usage,
                "output_tokens",
                None,
            )

        if "status" in df.columns:
            df.at[
                idx,
                "status"
            ] = "ok"

        if "error" in df.columns:
            df.at[
                idx,
                "error"
            ] = None

        print(
            f"\n{row['sample_id']}: "
            f"OK | {latency:.2f}s | "
            f"{answer[:100]}"
        )

        # Save after every successful case.
        df.to_csv(
            INPUT,
            index=False,
        )

    except Exception as e:

        print(
            f"\n{row['sample_id']}: "
            f"ERROR | "
            f"{type(e).__name__}: {e}"
        )


# Final check
remaining = df[
    df["large_answer"].isna()
    |
    (
        df["large_answer"]
        .astype(str)
        .str.strip()
        == ""
    )
]

print()
print("=" * 90)
print("DONE")
print("=" * 90)
print("Remaining missing:", len(remaining))

if len(remaining):
    print(
        remaining[
            [
                "sample_id",
                "question",
            ]
        ].to_string(
            index=False
        )
    )

print()
print("Updated:")
print(INPUT)
