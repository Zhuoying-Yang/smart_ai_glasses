import time
from pathlib import Path

from PIL import Image
from google import genai
from google.genai import types


MODEL = "gemini-3.7-flash"


client = genai.Client(
    http_options=types.HttpOptions(
        # 20 seconds per HTTP request
        timeout=20_000,

        # Do not let the SDK silently retry for a long time.
        retry_options=types.HttpRetryOptions(
            attempts=2,
            initial_delay=0.5,
            max_delay=2.0,
            exp_base=2.0,
            jitter=0.2,
            http_status_codes=[
                408,
                429,
                500,
                502,
                503,
                504,
            ],
        ),
    )
)


def ask_gemini_large(
    image_path: str,
    question: str,
):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    prompt = (
        question
        + "\nAnswer in English in one short sentence. "
          "Be concise and do not use bullet points. "
          "Answer based only on the provided image."
    )

    with Image.open(image_path) as img:
        image = img.convert("RGB")

        start = time.perf_counter()

        response = client.models.generate_content(
            model=MODEL,
            contents=[
                prompt,
                image,
            ],
        )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000.0

    answer = (
        response.text.strip()
        if response.text
        else ""
    )

    return answer, latency_ms
