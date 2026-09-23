from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CVFastDecision:
    eligible: bool
    target: str | None
    accepted: bool
    max_confidence: float
    threshold: float
    latency_ms: float
    reason: str


class CVFastPresence:
    """
    Positive-only YOLOE object-presence fast path.

    It only answers when confidence is high.
    Low-confidence cases fall back to the normal VLM router.
    """

    def __init__(
        self,
        threshold: float = 0.20,
        model_name: str | None = None,
        device: str | None = None,
    ):
        self.threshold = float(threshold)

        self.model_name = (
            model_name
            or os.environ.get(
                "YOLOE_FAST_MODEL",
                "yoloe-26s-seg.pt",
            )
        )

        self.device = (
            device
            or os.environ.get(
                "YOLOE_FAST_DEVICE"
            )
        )

        self._model = None
        self._load_error = None

    @staticmethod
    def _extract_target(
        question: str,
    ) -> str | None:

        q = str(
            question or ""
        ).strip().lower()

        q = re.sub(
            r"\s+",
            " ",
            q,
        )

        patterns = [
            (
                r"^(?:do|can) you see\s+"
                r"(?P<target>.+?)[?.!]*$"
            ),
            (
                r"^(?:is|are) there\s+"
                r"(?P<target>.+?)[?.!]*$"
            ),
        ]

        target = None

        for pattern in patterns:
            match = re.match(
                pattern,
                q,
            )

            if match:
                target = (
                    match
                    .group("target")
                    .strip()
                )
                break

        if not target:
            return None

        target = re.sub(
            r"^(?:a|an|any|the|some|my)\s+",
            "",
            target,
        )

        target = re.split(
            (
                r"\s+(?:"
                r"in front of me|"
                r"around me|"
                r"near me|"
                r"nearby|"
                r"here|"
                r"anywhere|"
                r"in this (?:image|frame|view)|"
                r"on the|"
                r"in the|"
                r"at the|"
                r"near the|"
                r"beside the|"
                r"behind the|"
                r"under the|"
                r"above the"
                r")\b"
            ),
            target,
            maxsplit=1,
        )[0].strip()

        if not target:
            return None

        if len(
            target.split()
        ) > 5:
            return None

        if any(
            token in target
            for token in [
                " or ",
                " and ",
            ]
        ):
            return None

        return target

    def _resolve_device(self):

        if self.device:
            return self.device

        try:
            import torch

            if (
                torch.backends
                .mps
                .is_available()
            ):
                return "mps"

        except Exception:
            pass

        return "cpu"

    def _load(self):

        if (
            self._model is not None
            or self._load_error is not None
        ):
            return

        try:
            from ultralytics import YOLOE

            self._model = YOLOE(
                self.model_name
            )

        except Exception as exc:
            self._load_error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    def try_presence(
        self,
        question: str,
        image_path: str | Path,
    ) -> CVFastDecision:

        target = (
            self._extract_target(
                question
            )
        )

        if target is None:
            return CVFastDecision(
                eligible=False,
                target=None,
                accepted=False,
                max_confidence=0.0,
                threshold=self.threshold,
                latency_ms=0.0,
                reason=(
                    "not_simple_presence_question"
                ),
            )

        image_path = Path(
            image_path
        )

        if not image_path.exists():
            return CVFastDecision(
                eligible=True,
                target=target,
                accepted=False,
                max_confidence=0.0,
                threshold=self.threshold,
                latency_ms=0.0,
                reason="image_missing",
            )

        self._load()

        if self._model is None:
            return CVFastDecision(
                eligible=True,
                target=target,
                accepted=False,
                max_confidence=0.0,
                threshold=self.threshold,
                latency_ms=0.0,
                reason=(
                    "yoloe_unavailable: "
                    f"{self._load_error}"
                ),
            )

        start = time.perf_counter()

        try:

            self._model.set_classes(
                [target]
            )

            result = (
                self._model.predict(
                    source=str(
                        image_path
                    ),
                    device=(
                        self._resolve_device()
                    ),
                    conf=0.01,
                    verbose=False,
                )[0]
            )

            confidences = []

            if (
                result.boxes
                is not None
                and result.boxes.conf
                is not None
            ):
                confidences = (
                    result
                    .boxes
                    .conf
                    .detach()
                    .cpu()
                    .tolist()
                )

            max_conf = (
                max(confidences)
                if confidences
                else 0.0
            )

            accepted = (
                max_conf
                >= self.threshold
            )

            return CVFastDecision(
                eligible=True,
                target=target,
                accepted=accepted,
                max_confidence=float(
                    max_conf
                ),
                threshold=self.threshold,
                latency_ms=(
                    (
                        time.perf_counter()
                        - start
                    )
                    * 1000.0
                ),
                reason=(
                    "high_confidence_positive"
                    if accepted
                    else
                    "low_confidence_fallback"
                ),
            )

        except Exception as exc:

            return CVFastDecision(
                eligible=True,
                target=target,
                accepted=False,
                max_confidence=0.0,
                threshold=self.threshold,
                latency_ms=(
                    (
                        time.perf_counter()
                        - start
                    )
                    * 1000.0
                ),
                reason=(
                    "yoloe_error: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )
