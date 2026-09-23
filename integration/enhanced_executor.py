from __future__ import annotations

import re
import time
from pathlib import Path

from PIL import Image, ImageDraw

from integration.final_executor import (
    FinalExecutor,
    FRAME_PATH,
)

from routing.cv_fast_presence import (
    CVFastPresence,
)

from routing.phone_bridge import (
    ask_phone,
)


TEMPORAL_SHEET_PATH = Path(
    "/tmp/aria_temporal_contact_sheet.jpg"
)

KNOWLEDGE_BLANK_PATH = Path(
    "/tmp/aria_knowledge_blank.jpg"
)


LARGE_TEMPORAL_PATTERNS = [
    r"\bfirst\b",
    r"\bbefore\b",
    r"\bafter\b",
    r"\bearlier\b",
    r"\blater\b",
    r"\border\b",
    r"\bsequence\b",
    r"\bhow many times\b",
]


SMALL_TEMPORAL_PATTERNS = [
    r"\bwhat happened\b",
    r"^did\b",
    r"^was\b",
    r"^were\b",
    r"^has\b",
    r"^have\b",
]


LIVE_KNOWLEDGE_PATTERNS = [
    r"\bcurrent(?:ly)?\b",
    r"\bright now\b",
    r"\bnow\b",
    r"\btoday\b",
    r"\btonight\b",
    r"\btomorrow\b",
    r"\blatest\b",
    r"\brecent\b",
    r"\bnews\b",
    r"\bweather\b",
    r"\bforecast\b",
    r"\btemperature\b",
    r"\bopen now\b",
    r"\bopening hours?\b",
    r"\bclosing time\b",
    r"\bnear me\b",
    r"\bnearby\b",
    r"\bclosest\b",
    r"\btraffic\b",
    r"\bdelay(?:ed|s)?\b",
    r"\bschedule\b",
    r"\bscore\b",
    r"\bstock price\b",
    r"\bexchange rate\b",
]


class EnhancedFinalExecutor(
    FinalExecutor
):

    def __init__(
        self,
        visual_buffer,
        speak=True,
    ):

        super().__init__(
            visual_buffer=
                visual_buffer,
            speak=speak,
        )

        self.cv_fast = (
            CVFastPresence(
                threshold=0.20
            )
        )

    @staticmethod
    def _use_small_temporal(
        question,
    ):

        q = str(
            question or ""
        ).strip().lower()

        if any(
            re.search(
                pattern,
                q,
            )
            for pattern
            in LARGE_TEMPORAL_PATTERNS
        ):
            return False

        return any(
            re.search(
                pattern,
                q,
            )
            for pattern
            in SMALL_TEMPORAL_PATTERNS
        )

    @staticmethod
    def _knowledge_needs_web(
        question,
    ):

        q = str(
            question or ""
        ).strip().lower()

        return any(
            re.search(
                pattern,
                q,
            )
            for pattern
            in LIVE_KNOWLEDGE_PATTERNS
        )

    @staticmethod
    def _normalize_phone_answer(
        result,
    ):

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

        return str(
            answer or ""
        ).strip()

    def _ask_small_image(
        self,
        image_path,
        prompt,
        mode,
    ):

        print(
            "Backend : SMALL"
        )

        print(
            "Model   : Gemma E2B on Galaxy"
        )

        print(
            "Mode    :",
            mode,
        )

        start = (
            time.perf_counter()
        )

        result = ask_phone(
            str(image_path),
            prompt,
            timeout=30,
        )

        latency = (
            time.perf_counter()
            - start
        )

        answer = (
            self
            ._normalize_phone_answer(
                result
            )
        )

        print(
            "Answer  :",
            answer,
        )

        print(
            "Latency :",
            f"{latency:.2f} s",
        )

        return answer

    def _build_temporal_contact_sheet(
        self,
        seconds=12.0,
        k=8,
    ):

        frames = (
            self
            .visual_buffer
            .uniform_sample(
                seconds=seconds,
                k=k,
            )
        )

        if len(frames) < 2:
            return None, 0

        tile_size = 320
        label_height = 26
        columns = 4

        rows = (
            (
                len(frames)
                + columns
                - 1
            )
            // columns
        )

        sheet = Image.new(
            "RGB",
            (
                columns
                * tile_size,
                rows
                * (
                    tile_size
                    + label_height
                ),
            ),
            "white",
        )

        draw = (
            ImageDraw.Draw(
                sheet
            )
        )

        for i, frame in enumerate(
            frames
        ):

            with Image.open(
                frame.path
            ) as img:

                img = (
                    img.convert(
                        "RGB"
                    )
                )

                img.thumbnail(
                    (
                        tile_size,
                        tile_size,
                    ),
                    Image.Resampling.LANCZOS,
                )

                tile = Image.new(
                    "RGB",
                    (
                        tile_size,
                        tile_size,
                    ),
                    "black",
                )

                x0 = (
                    tile_size
                    - img.width
                ) // 2

                y0 = (
                    tile_size
                    - img.height
                ) // 2

                tile.paste(
                    img,
                    (
                        x0,
                        y0,
                    ),
                )

            row = (
                i
                // columns
            )

            col = (
                i
                % columns
            )

            x = (
                col
                * tile_size
            )

            y = (
                row
                * (
                    tile_size
                    + label_height
                )
            )

            sheet.paste(
                tile,
                (
                    x,
                    y
                    + label_height,
                ),
            )

            draw.text(
                (
                    x + 6,
                    y + 6,
                ),
                f"Frame {i + 1}",
                fill="black",
            )

        sheet.save(
            TEMPORAL_SHEET_PATH,
            format="JPEG",
            quality=88,
        )

        return (
            TEMPORAL_SHEET_PATH,
            len(frames),
        )

    @staticmethod
    def _ensure_knowledge_blank():

        if (
            not
            KNOWLEDGE_BLANK_PATH.exists()
        ):

            Image.new(
                "RGB",
                (
                    512,
                    512,
                ),
                (
                    220,
                    220,
                    220,
                ),
            ).save(
                KNOWLEDGE_BLANK_PATH,
                format="JPEG",
                quality=90,
            )

        return (
            KNOWLEDGE_BLANK_PATH
        )

    def _direct_visual(
        self,
        question,
        frame_rgb,
    ):

        self._save_frame(
            frame_rgb
        )

        cv_decision = (
            self
            .cv_fast
            .try_presence(
                question=
                    question,
                image_path=
                    FRAME_PATH,
            )
        )

        if cv_decision.eligible:

            print()
            print(
                "CV FAST PRESENCE"
            )

            print(
                "Target  :",
                cv_decision.target,
            )

            print(
                "YOLO conf:",
                (
                    f"{cv_decision.max_confidence:.3f}"
                ),
            )

            print(
                "Threshold:",
                (
                    f"{cv_decision.threshold:.3f}"
                ),
            )

            print(
                "Latency :",
                (
                    f"{cv_decision.latency_ms:.1f} ms"
                ),
            )

            print(
                "Decision:",
                cv_decision.reason,
            )

            if cv_decision.accepted:

                answer = (
                    "Yes. I can see "
                    f"{cv_decision.target}."
                )

                print(
                    "Answer  :",
                    answer,
                )

                return answer

            print(
                "Fallback: normal "
                "SMALL/LARGE visual router"
            )

        return (
            super()
            ._direct_visual(
                question,
                frame_rgb,
            )
        )

    def _temporal(
        self,
        question,
    ):

        if not (
            self
            ._use_small_temporal(
                question
            )
        ):

            return (
                super()
                ._temporal(
                    question
                )
            )

        print()
        print(
            "TEMPORAL EXECUTION"
        )

        print(
            "Policy  : "
            "SMALL temporal contact sheet"
        )

        (
            sheet_path,
            num_frames,
        ) = (
            self
            ._build_temporal_contact_sheet(
                seconds=12.0,
                k=8,
            )
        )

        if sheet_path is None:

            print(
                "Fallback: not enough "
                "frames for SMALL temporal"
            )

            return (
                super()
                ._temporal(
                    question
                )
            )

        prompt = (
            f"The image contains {num_frames} "
            "recent video frames in chronological "
            "order. Read left-to-right across the "
            "top row, then continue left-to-right "
            "on the next row. Frame 1 is the oldest "
            "and the last frame is the newest.\n\n"
            f"Question: {question}\n\n"
            "Reason across the frames and answer "
            "directly in one concise sentence. "
            "Only describe what is clearly visible."
        )

        try:

            return (
                self
                ._ask_small_image(
                    sheet_path,
                    prompt,
                    mode=(
                        "TEMPORAL_CONTACT_SHEET"
                    ),
                )
            )

        except Exception as exc:

            print(
                "SMALL temporal failed; "
                "falling back to LARGE:",
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

            return (
                super()
                ._temporal(
                    question
                )
            )

    def _knowledge(
        self,
        question,
        frame_rgb,
    ):

        if (
            self
            ._knowledge_needs_web(
                question
            )
        ):

            print()
            print(
                "KNOWLEDGE POLICY: "
                "live/current info "
                "-> WEB + LARGE"
            )

            return (
                super()
                ._knowledge(
                    question,
                    frame_rgb,
                )
            )

        print()
        print(
            "KNOWLEDGE POLICY: "
            "static/common knowledge "
            "-> SMALL"
        )

        prompt = (
            "Ignore the image. Answer the "
            "following general knowledge "
            "question from your own knowledge. "
            "Give a short, direct answer in "
            "one or two sentences. "
            "Do not mention the image.\n\n"
            f"Question: {question}"
        )

        try:

            return (
                self
                ._ask_small_image(
                    (
                        self
                        ._ensure_knowledge_blank()
                    ),
                    prompt,
                    mode=(
                        "STATIC_KNOWLEDGE"
                    ),
                )
            )

        except Exception as exc:

            print(
                "SMALL knowledge failed; "
                "falling back to WEB + LARGE:",
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

            return (
                super()
                ._knowledge(
                    question,
                    frame_rgb,
                )
            )
