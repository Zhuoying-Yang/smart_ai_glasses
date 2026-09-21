from pathlib import Path
import base64
import mimetypes
import os
import subprocess
import sys
import time

import cv2
from openai import OpenAI

from routing.temporal_memory import execute_temporal
from routing.visual_rescue_router import VisualRescueRouter
from integration.audio_state import begin_tts, end_tts


OLD_PROJECT = (
    Path.home()
    / "projectaria_client_sdk_samples"
)

FRAME_PATH = Path(
    "/tmp/aria_router_frame.jpg"
)

LARGE_MODEL = os.environ.get(
    "OPENAI_LARGE_MODEL",
    os.environ.get(
        "LARGE_VLM_MODEL",
        "gpt-5.6-luna",
    ),
)


def image_to_data_url(
    path: Path,
) -> str:

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


class FinalExecutor:
    def __init__(
        self,
        visual_buffer,
        speak=True,
    ):
        self.visual_buffer = (
            visual_buffer
        )

        self.speak = speak

        self.visual_router = (
            VisualRescueRouter()
        )

        self._large_client = None

    def _speak(
        self,
        answer,
    ):
        answer = str(
            answer or ""
        ).strip()

        if self.speak and answer:
            begin_tts()

            try:
                subprocess.run(
                    [
                        "say",
                        answer,
                    ],
                    check=False,
                )

            finally:
                end_tts(
                    grace_seconds=1.2
                )

    def _save_frame(
        self,
        frame_rgb,
    ):
        if frame_rgb is None:
            raise ValueError(
                "No current RGB frame."
            )

        frame_bgr = cv2.cvtColor(
            frame_rgb,
            cv2.COLOR_RGB2BGR,
        )

        ok = cv2.imwrite(
            str(FRAME_PATH),
            frame_bgr,
        )

        if not ok:
            raise RuntimeError(
                "Failed to save Aria frame."
            )

    def _client(self):
        if self._large_client is None:
            self._large_client = (
                OpenAI()
            )

        return self._large_client

    def _small(
        self,
        question,
    ):
        from routing.phone_bridge import ask_phone

        prompt = (
            f"{question}\n"
            "Answer directly in one concise sentence."
        )

        print("Backend : SMALL")
        print("Model   : Gemma E2B on Galaxy")

        t0 = time.perf_counter()

        result = ask_phone(
            str(FRAME_PATH),
            prompt,
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

        print("Answer  :", answer)
        print(
            "Latency :",
            f"{latency:.2f} s",
        )

        return answer

    def _large_visual(
        self,
        question,
    ):
        prompt = (
            f"{question}\n"
            "Answer directly in one concise sentence. "
            "Use the current wearable-camera image."
        )

        print("Backend : LARGE")
        print("Model   :", LARGE_MODEL)

        t0 = time.perf_counter()

        response = (
            self._client()
            .responses.create(
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
                max_output_tokens=160,
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

        print("Answer  :", answer)
        print(
            "Latency :",
            f"{latency:.2f} s",
        )

        return answer

    def _direct_visual(
        self,
        question,
        frame_rgb,
    ):
        decision = (
            self.visual_router.route(
                question
            )
        )

        print()
        print(
            "DIRECT_VISUAL MODEL ROUTER"
        )
        print(
            "Rescue P:",
            f"{decision.rescue_probability:.3f}",
        )
        print(
            "Threshold:",
            f"{decision.threshold:.3f}",
        )
        print(
            "Chosen  :",
            decision.backend,
        )

        self._save_frame(
            frame_rgb
        )

        if (
            decision.backend
            == "LARGE"
        ):
            return self._large_visual(
                question
            )

        return self._small(
            question
        )

    def _temporal(
        self,
        question,
    ):
        print()
        print("TEMPORAL EXECUTION")
        print(
            "History : last ~12 s"
        )
        print(
            "Frames  : up to 4"
        )

        result = execute_temporal(
            question=question,
            visual_buffer=
                self.visual_buffer,
            seconds=12.0,
            num_frames=4,
        )

        answer = result[
            "answer"
        ]

        if result.get(
            "success"
        ):
            print(
                "Model   :",
                result.get(
                    "model"
                ),
            )
            print(
                "Frames  :",
                result.get(
                    "num_frames"
                ),
            )
            print(
                "Cloud latency:",
                f"{result.get('cloud_latency', 0):.2f} s",
            )
            print(
                "E2E latency  :",
                f"{result.get('e2e_latency', 0):.2f} s",
            )

        print(
            "Answer  :",
            answer,
        )

        return answer

    def _knowledge(
        self,
        question,
        frame_rgb,
    ):
        print()
        print("KNOWLEDGE EXECUTION")
        print("Backend : WEB_SEARCH + LARGE")
        print("Model   :", LARGE_MODEL)

        content = [
            {
                "type": "input_text",
                "text": (
                    "You are answering a question "
                    "from wearable AI glasses. "
                    "Use web search when needed. "
                    "Give a short answer suitable "
                    "for speaking aloud.\n\n"
                    f"Question: {question}"
                ),
            }
        ]

        if frame_rgb is not None:

            self._save_frame(
                frame_rgb
            )

            content.append(
                {
                    "type":
                    "input_image",
                    "image_url":
                    image_to_data_url(
                        FRAME_PATH
                    ),
                    "detail":
                    "auto",
                }
            )

            print(
                "Visual context: current Aria frame"
            )

        t0 = time.perf_counter()

        response = (
            self._client()
            .responses.create(
                model=LARGE_MODEL,
                tools=[
                    {
                        "type":
                        "web_search"
                    }
                ],
                input=[
                    {
                        "role": "user",
                        "content":
                            content,
                    }
                ],
                max_output_tokens=180,
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

        print(
            "Answer  :",
            answer,
        )
        print(
            "Latency :",
            f"{latency:.2f} s",
        )

        return answer

    def execute_proactive(
        self,
        event,
        frame_rgb,
    ):
        kind = event.trigger_type.value
        description = str(
            event.query.text
        ).strip()

        if kind == "ALERT":
            answer = (
                "Warning. I detected "
                + description
                + "."
            )
        else:
            answer = (
                "Reminder. I detected "
                + description
                + "."
            )

        print()
        print("-" * 68)
        print("PROACTIVE EXECUTOR")
        print("-" * 68)
        print("Trigger :", kind)
        print("Detected:", description)
        print("Score   :", f"{event.score:.3f}")
        print("Threshold:", f"{event.threshold:.3f}")
        print("Answer  :", answer)
        print("-" * 68)
        print()

        self._speak(answer)

        return answer

    def execute(
        self,
        decision,
        event,
        frame_rgb,
    ):
        question = (
            event.query.text
        )

        route = (
            decision.route
        )

        print()
        print("-" * 68)
        print("FINAL EXECUTOR")
        print("-" * 68)
        print("Route   :", route)

        try:

            if route == "DIRECT_VISUAL":
                answer = (
                    self._direct_visual(
                        question,
                        frame_rgb,
                    )
                )

            elif route == "TEMPORAL":
                answer = (
                    self._temporal(
                        question
                    )
                )

            elif route == "KNOWLEDGE":
                answer = (
                    self._knowledge(
                        question,
                        frame_rgb,
                    )
                )

            elif route == "MEMORY":
                answer = (
                    "Long-term memory module "
                    "is not connected yet."
                )

                print(
                    "Status  : waiting for "
                    "memory-team integration"
                )

            else:
                raise ValueError(
                    f"Unknown route: {route}"
                )

        except Exception as exc:

            print(
                "ERROR   :",
                f"{type(exc).__name__}: "
                f"{exc}",
            )

            print("-" * 68)

            return ""

        print("-" * 68)
        print()

        self._speak(
            answer
        )

        return answer
