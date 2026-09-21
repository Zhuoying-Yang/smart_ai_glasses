from __future__ import annotations

import csv
import json
import os
import re
import time
from pathlib import Path

from routing.large_multiframe import call_large_multiframe


LOG_PATH = Path(
    "routing/benchmarks/results/short_term_routing_log.csv"
)

LOG_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


def log_result(
    route,
    question,
    frames,
    result,
    e2e_latency,
):
    row = {
        "timestamp": time.time(),
        "route": route,
        "question": question,
        "num_frames": len(frames),
        "frame_ids": json.dumps(
            [f.frame_id for f in frames]
        ),
        "frame_timestamps": json.dumps(
            [
                round(f.timestamp, 3)
                for f in frames
            ]
        ),
        "model": result["model"],
        "cloud_latency_s": round(
            result["cloud_latency"],
            4,
        ),
        "e2e_latency_s": round(
            e2e_latency,
            4,
        ),
        "answer": result["answer"],
    }

    write_header = not LOG_PATH.exists()

    with LOG_PATH.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys()),
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)


ORDERING_PATTERNS = [
    r"\bfirst\b",
    r"\bbefore\b",
    r"\bafter\b",
    r"\bearlier\b",
    r"\blater\b",
    r"\border\b",
]


def is_ordering_question(question):
    q = str(question).lower()

    return any(
        re.search(pattern, q)
        for pattern in ORDERING_PATTERNS
    )


def execute_temporal(
    question,
    visual_buffer,
    seconds=12.0,
    num_frames=4,
):
    start = time.perf_counter()

    selector = os.getenv(
        "TEMPORAL_SELECTOR",
        "uniform",
    ).lower()

    ordering = is_ordering_question(
        question
    )

    if ordering:
        effective_frames = 6
        frames = visual_buffer.uniform_sample(
            seconds=seconds,
            k=effective_frames,
        )
        selector_used = "uniform-ordering"

    elif selector == "motion":
        effective_frames = num_frames
        frames = visual_buffer.motion_sample(
            seconds=seconds,
            k=effective_frames,
        )
        selector_used = "motion"

    else:
        effective_frames = num_frames
        frames = visual_buffer.uniform_sample(
            seconds=seconds,
            k=effective_frames,
        )
        selector_used = "uniform"

    print(
        "[TEMPORAL] selector:",
        selector_used,
    )

    if len(frames) < 2:
        return {
            "route": "TEMPORAL",
            "answer": (
                "I don't have enough recent "
                "visual history yet."
            ),
            "success": False,
            "num_frames": len(frames),
        }

    print()
    print(
        f"[TEMPORAL] selected {len(frames)} frames"
    )
    print(
        "[TEMPORAL] frame ages:",
        [round(f.age(), 1) for f in frames],
    )

    result = call_large_multiframe(
        question=question,
        frames=frames,
        mode="TEMPORAL",
    )

    e2e_latency = time.perf_counter() - start

    log_result(
        route="TEMPORAL",
        question=question,
        frames=frames,
        result=result,
        e2e_latency=e2e_latency,
    )

    return {
        "route": "TEMPORAL",
        "answer": result["answer"],
        "success": True,
        "num_frames": len(frames),
        "frame_ids": [
            f.frame_id for f in frames
        ],
        "cloud_latency": result["cloud_latency"],
        "e2e_latency": e2e_latency,
        "model": result["model"],
    }


def execute_memory(
    question,
    visual_buffer,
    seconds=15.0,
    num_frames=8,
):
    start = time.perf_counter()

    frames = visual_buffer.uniform_sample(
        seconds=seconds,
        k=num_frames,
    )

    if not frames:
        return {
            "route": "MEMORY",
            "answer": (
                "I don't have any recent "
                "visual memory yet."
            ),
            "success": False,
            "num_frames": 0,
        }

    print()
    print(
        f"[MEMORY] selected {len(frames)} frames"
    )
    print(
        "[MEMORY] frame ages:",
        [round(f.age(), 1) for f in frames],
    )

    result = call_large_multiframe(
        question=question,
        frames=frames,
        mode="MEMORY",
    )

    e2e_latency = time.perf_counter() - start

    log_result(
        route="MEMORY",
        question=question,
        frames=frames,
        result=result,
        e2e_latency=e2e_latency,
    )

    return {
        "route": "MEMORY",
        "answer": result["answer"],
        "success": True,
        "num_frames": len(frames),
        "frame_ids": [
            f.frame_id for f in frames
        ],
        "cloud_latency": result["cloud_latency"],
        "e2e_latency": e2e_latency,
        "model": result["model"],
    }
