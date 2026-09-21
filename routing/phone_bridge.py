from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


PACKAGE = "com.zhuoying.ariaedgevlm"
ACTIVITY = f"{PACKAGE}/.MainActivity"
ACTION = f"{PACKAGE}.RUN_BRIDGE"

SERIAL = os.environ.get(
    "ANDROID_SERIAL",
    "R3GL70M9Y4N",
)

ADB = [
    "adb",
    "-s",
    SERIAL,
]


def run(args, capture=False, check=True):
    cmd = ADB + list(args)

    print(
        "+",
        " ".join(cmd),
    )

    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=capture,
    )


def _extract_answer(raw):
    raw = str(raw).strip()

    if not raw:
        return ""

    try:
        obj = json.loads(raw)
    except Exception:
        return raw

    if isinstance(obj, dict):
        for key in [
            "answer",
            "response",
            "text",
            "output",
            "prediction",
            "result",
        ]:
            value = obj.get(key)

            if value is not None:
                return str(value).strip()

    return raw


def ask_phone(
    image_path,
    prompt,
    timeout=30,
):
    image_path = (
        Path(image_path)
        .expanduser()
        .resolve()
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(prompt)
        prompt_path = Path(f.name)

    try:
        run([
            "push",
            str(image_path),
            "/data/local/tmp/input.jpg",
        ])

        run([
            "push",
            str(prompt_path),
            "/data/local/tmp/prompt.txt",
        ])

        run([
            "shell",
            "run-as",
            PACKAGE,
            "mkdir",
            "-p",
            "files/bridge",
        ])

        run([
            "shell",
            "run-as",
            PACKAGE,
            "cp",
            "/data/local/tmp/input.jpg",
            "files/bridge/input.jpg",
        ])

        run([
            "shell",
            "run-as",
            PACKAGE,
            "cp",
            "/data/local/tmp/prompt.txt",
            "files/bridge/prompt.txt",
        ])

        subprocess.run(
            ADB + [
                "shell",
                "run-as",
                PACKAGE,
                "rm",
                "-f",
                "files/bridge/output.txt",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        print()
        print(
            "Triggering Galaxy E2B inference..."
        )
        print()

        run([
            "shell",
            "am",
            "start",
            "-n",
            ACTIVITY,
            "-a",
            ACTION,
        ])

        start = time.time()

        while True:
            check = subprocess.run(
                ADB + [
                    "shell",
                    "run-as",
                    PACKAGE,
                    "test",
                    "-f",
                    "files/bridge/output.txt",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            if check.returncode == 0:
                time.sleep(0.15)

                result = run(
                    [
                        "shell",
                        "run-as",
                        PACKAGE,
                        "cat",
                        "files/bridge/output.txt",
                    ],
                    capture=True,
                )

                raw = result.stdout.strip()

                if not raw:
                    raise RuntimeError(
                        "Galaxy created output.txt "
                        "but it was empty."
                    )

                return _extract_answer(
                    raw
                )

            if (
                time.time()
                - start
                > timeout
            ):
                raise TimeoutError(
                    "Galaxy E2B inference "
                    f"timed out after {timeout}s."
                )

            time.sleep(0.1)

    finally:
        try:
            prompt_path.unlink()
        except FileNotFoundError:
            pass
