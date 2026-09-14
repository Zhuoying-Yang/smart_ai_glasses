from pathlib import Path
import subprocess
import sys
import time

import cv2

from routing.action_space import Action


# Reuse the already-working Galaxy bridge.
OLD_PROJECT = Path.home() / "projectaria_client_sdk_samples"
sys.path.insert(0, str(OLD_PROJECT))

from phone_bridge import ask_phone


FRAME_PATH = Path("/tmp/aria_router_frame.jpg")


class WhichExecutor:
    def __init__(self, speak=True):
        self.speak = speak

    def execute(self, decision, event, frame_rgb):
        print()
        print("-" * 65)
        print("EXECUTOR")
        print("-" * 65)

        # --------------------------------------------------
        # LARGE is intentionally blank for now.
        # --------------------------------------------------
        if decision.action in (
            Action.LARGE_1F,
            Action.LARGE_MULTI,
        ):
            print("Backend : LARGE")
            print("Status  : backend not connected yet")
            print("Answer  :")
            print("-" * 65)
            return ""

        # --------------------------------------------------
        # SMALL -> Galaxy Gemma E2B
        # --------------------------------------------------
        if decision.action not in (
            Action.SMALL_1F,
            Action.SMALL_MULTI,
        ):
            print("No model execution for:", decision.action.value)
            print("-" * 65)
            return ""

        print("Backend : SMALL")
        print("Model   : Gemma E2B on Galaxy")

        if frame_rgb is None:
            print("ERROR   : no RGB frame available")
            print("-" * 65)
            return ""

        # WHEN uses RGB; cv2.imwrite expects BGR.
        frame_bgr = cv2.cvtColor(
            frame_rgb,
            cv2.COLOR_RGB2BGR,
        )

        if not cv2.imwrite(str(FRAME_PATH), frame_bgr):
            raise RuntimeError("Failed to save current Aria frame")

        question = event.query.text

        # Keep the original question for WHICH classification,
        # but ask the local VLM to answer briefly.
        phone_prompt = (
            f"{question}\n"
            "Answer directly in one concise sentence. "
            "Do not provide a list or additional explanation."
        )

        print("Question:", question)
        print("Sending current Aria frame to Galaxy...")

        t0 = time.perf_counter()

        result = ask_phone(
            str(FRAME_PATH),
            phone_prompt,
            timeout=30,
        )

        latency = time.perf_counter() - t0

        # Tolerate several bridge return formats.
        if isinstance(result, tuple):
            answer = result[0]
        elif isinstance(result, dict):
            answer = result.get("answer", str(result))
        else:
            answer = result

        answer = str(answer).strip()

        print()
        print("Answer  :", answer)
        print("Latency :", f"{latency:.2f} s")
        print("-" * 65)
        print()

        # Aria Gen1 has no speaker, so Mac speaks for now.
        if self.speak and answer:
            subprocess.run(
                ["say", answer],
                check=False,
            )

        return answer
