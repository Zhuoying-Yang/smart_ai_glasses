import os
import base64
import mimetypes
import time
from pathlib import Path

import requests


MODEL = "google/gemma-4-31b-it:free"

API_URL = "https://openrouter.ai/api/v1/chat/completions"


def encode_image(image_path):
    mime_type, _ = mimetypes.guess_type(str(image_path))

    if mime_type is None:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(
            f.read()
        ).decode("utf-8")

    return (
        f"data:{mime_type};"
        f"base64,{encoded}"
    )


def ask_openrouter_large(
    image_path: str,
    question: str,
):
    api_key = os.environ.get(
        "OPENROUTER_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set."
        )

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            image_path
        )

    image_data = encode_image(
        image_path
    )

    prompt = (
        question
        + "\nAnswer in English in one short sentence. "
          "Be concise and do not use bullet points. "
          "Answer based only on the provided image."
    )

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data,
                        },
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 80,
    }

    headers = {
        "Authorization": (
            f"Bearer {api_key}"
        ),
        "Content-Type": "application/json",
    }

    start = time.perf_counter()

    response = None

    for attempt in range(3):
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )

        if response.status_code != 429:
            break

        # Small controlled backoff for free-tier rate limiting.
        wait_s = 5 * (attempt + 1)
        print(
            f"OpenRouter rate limited (429). "
            f"Retrying in {wait_s}s..."
        )
        time.sleep(wait_s)

    latency_ms = (
        time.perf_counter() - start
    ) * 1000.0

    response.raise_for_status()

    data = response.json()

    answer = (
        data["choices"][0]
        ["message"]["content"]
        .strip()
    )

    return answer, latency_ms
