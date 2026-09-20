from pathlib import Path
import base64
import mimetypes
import os
import subprocess
import sys
import time

import cv2
from openai import OpenAI

from routing.action_space import Action


# ============================================================
# Galaxy SMALL backend
# ============================================================

OLD_PROJECT = (
    Path.home()
    / "projectaria_client_sdk_samples"
)

sys.path.insert(
    0,
    str(OLD_PROJECT),
)

from phone_bridge import ask_phone


# ============================================================
# Paths / config
# ============================================================

FRAME_PATH = Path(
    "/tmp/aria_router_frame.jpg"
)

LARGE_MODEL = os.environ.get(
    "LARGE_VLM_MODEL",
    "gpt-5.6-sol",
)


def image_to_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(
        str(path)
    )

    if mime is None:
        mime = "image/jpeg"

    data = base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")

    return (
        f"data:{mime};base64,{data}"
    )


class WhichExecutor:

    def __init__(
        self,
        speak=True,
    ):
        self.speak = speak
        self._large_client = None

    # ========================================================
    # Helpers
    # ========================================================

    def _speak(self, answer: str):

        if self.speak and answer:
            subprocess.run(
                ["say", answer],
                check=False,
            )

    def _save_frame(
        self,
        frame_rgb,
    ):

        if frame_rgb is None:
            raise ValueError(
                "No RGB frame available."
            )

        # WHEN gives RGB.
        # cv2.imwrite expects BGR.
        frame_bgr = cv2.cvtColor(
            frame_rgb,
            cv2.COLOR_RGB2BGR,
        )

        if not cv2.imwrite(
            str(FRAME_PATH),
            frame_bgr,
        ):
            raise RuntimeError(
                "Failed to save current Aria frame"
            )

    def _get_large_client(self):

        if self._large_client is None:
            self._large_client = OpenAI()

        return self._large_client

    # ========================================================
    # SMALL 1F
    # ========================================================

    def _execute_small_1f(
        self,
        question,
    ):

        print("Backend : SMALL")
        print("Model   : Gemma E2B on Galaxy")

        phone_prompt = (
            f"{question}\n"
            "Answer directly in one concise sentence. "
            "Do not provide a list or additional explanation."
        )

        print("Question:", question)
        print(
            "Sending current Aria frame "
            "to Galaxy..."
        )

        t0 = time.perf_counter()

        result = ask_phone(
            str(FRAME_PATH),
            phone_prompt,
            timeout=30,
        )

        latency = (
            time.perf_counter()
            - t0
        )

        if isinstance(
            result,
            tuple,
        ):
            answer = result[0]

        elif isinstance(
            result,
            dict,
        ):
            answer = result.get(
                "answer",
                str(result),
            )

        else:
            answer = result

        answer = str(
            answer
        ).strip()

        print()
        print(
            "Answer  :",
            answer,
        )
        print(
            "Latency :",
            f"{latency:.2f} s",
        )

        return answer

    # ========================================================
    # LARGE 1F
    # ========================================================

    def _execute_large_1f(
        self,
        question,
    ):

        print("Backend : LARGE")
        print(
            "Model   :",
            LARGE_MODEL,
        )

        prompt = (
            f"{question}\n"
            "Answer directly in one concise sentence. "
            "Use only the visual information needed "
            "to answer the question. "
            "Do not provide unnecessary explanation."
        )

        print("Question:", question)
        print(
            "Sending current Aria frame "
            "to cloud LARGE VLM..."
        )

        client = (
            self._get_large_client()
        )

        t0 = time.perf_counter()

        response = (
            client.responses.create(
                model=LARGE_MODEL,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type":
                                "input_text",
                                "text":
                                prompt,
                            },
                            {
                                "type":
                                "input_image",
                                "image_url":
                                image_to_data_url(
                                    FRAME_PATH
                                ),
                                "detail":
                                "auto",
                            },
                        ],
                    }
                ],
            )
        )

        latency = (
            time.perf_counter()
            - t0
        )

        answer = (
            response.output_text
            or ""
        ).strip()

        print()
        print(
            "Answer  :",
            answer,
        )
        print(
            "Latency :",
            f"{latency:.2f} s",
        )

        return answer

    # ========================================================
    # Main executor
    # ========================================================

    def execute(
        self,
        decision,
        event,
        frame_rgb,
    ):

        print()
        print("-" * 65)
        print("EXECUTOR")
        print("-" * 65)

        action = decision.action

        # ----------------------------------------------------
        # MULTI paths are deliberately NOT faked with one frame.
        # ----------------------------------------------------

        if action == Action.SMALL_MULTI:

            print("Backend : SMALL")
            print("Mode    : MULTI")
            print(
                "Status  : multi-frame buffer "
                "not connected yet"
            )
            print("-" * 65)

            return ""

        if action == Action.LARGE_MULTI:

            print("Backend : LARGE")
            print("Mode    : MULTI")
            print(
                "Status  : multi-frame buffer "
                "not connected yet"
            )
            print("-" * 65)

            return ""

        # ----------------------------------------------------
        # No model action
        # ----------------------------------------------------

        if action not in (
            Action.SMALL_1F,
            Action.LARGE_1F,
        ):

            print(
                "No model execution for:",
                action.value,
            )

            print("-" * 65)

            return ""

        # ----------------------------------------------------
        # Both 1F backends use the current Aria RGB frame.
        # ----------------------------------------------------

        try:
            self._save_frame(
                frame_rgb
            )

        except Exception as e:

            print(
                "ERROR   :",
                str(e),
            )

            print("-" * 65)

            return ""

        question = (
            event.query.text
        )

        try:

            if (
                action
                == Action.SMALL_1F
            ):
                answer = (
                    self._execute_small_1f(
                        question
                    )
                )

            else:
                answer = (
                    self._execute_large_1f(
                        question
                    )
                )

        except Exception as e:

            print()
            print(
                "ERROR   :",
                f"{type(e).__name__}: {e}",
            )

            print("-" * 65)

            return ""

        print("-" * 65)
        print()

        self._speak(
            answer
        )

        return answer
