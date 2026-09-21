from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from io import BytesIO

from PIL import Image, ImageOps

from openai import OpenAI


def image_to_data_url(path, max_side=1024):
    """
    Resize image in memory before sending to cloud.

    Keeps aspect ratio and limits the longest side to max_side.
    The original buffered frame on disk is NOT modified.
    """

    with Image.open(path) as img:
        img.load()
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")

        width, height = img.size

        longest = max(width, height)

        if longest > max_side:
            scale = max_side / longest

            new_size = (
                max(1, round(width * scale)),
                max(1, round(height * scale)),
            )

            img = img.resize(
                new_size,
                Image.Resampling.LANCZOS,
            )

        buffer = BytesIO()

        img.save(
            buffer,
            format="JPEG",
            quality=85,
            optimize=True,
        )

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


def call_large_multiframe(
    question,
    frames,
    mode,
):
    if not frames:
        raise ValueError("No frames supplied.")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Run: export OPENAI_API_KEY='your-real-key'"
        )

    model = os.getenv(
        "OPENAI_LARGE_MODEL",
        "gpt-5.6-luna",
    )

    client = OpenAI()

    mode = mode.upper()

    if mode == "TEMPORAL":
        instructions = """
You are the visual reasoning component of wearable AI glasses.

The supplied images show the wearer's recent visual history.
They are ordered chronologically from oldest to newest.

Answer the user's TEMPORAL question by reasoning across the sequence.
Look for actions, movement, changes, ordering, and events.

Use only evidence visible in the supplied frames.
If there is not enough visual evidence, say so clearly.

Give a short answer suitable for speaking aloud.
""".strip()

    elif mode == "MEMORY":
        instructions = """
You are the short-term visual memory component of wearable AI glasses.

The supplied images are representative frames from approximately the
last 15 seconds of what the wearer saw.

They are ordered chronologically from oldest to newest.

Find the earlier frame or frames relevant to the user's question and
answer using that visual evidence.

Use only information visible in these frames.
If the requested information was not captured, say so clearly.

Give a short answer suitable for speaking aloud.
""".strip()

    else:
        raise ValueError(
            f"Unsupported mode: {mode}"
        )

    now = time.time()

    content = [
        {
            "type": "input_text",
            "text": instructions,
        }
    ]

    for i, frame in enumerate(frames):
        age = max(
            0.0,
            now - frame.timestamp,
        )

        content.append(
            {
                "type": "input_text",
                "text": (
                    f"Frame {i + 1}/{len(frames)}, "
                    f"approximately {age:.1f} seconds ago:"
                ),
            }
        )

        content.append(
            {
                "type": "input_image",
                "image_url": image_to_data_url(
                    frame.path
                ),
            }
        )

    content.append(
        {
            "type": "input_text",
            "text": (
                "User question:\n"
                f"{question}"
            ),
        }
    )

    start = time.perf_counter()

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": content,
            }
        ],
        max_output_tokens=200,
    )

    latency = time.perf_counter() - start

    return {
        "answer": response.output_text.strip(),
        "model": model,
        "cloud_latency": latency,
        "num_frames": len(frames),
    }
