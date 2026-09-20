from pathlib import Path
import json

import pandas as pd


ROOT = Path(
    "routing/benchmarks/egowearbench/raw"
)

OUT_QUESTIONS = Path(
    "routing/benchmarks/egowearbench/"
    "question_manifest.csv"
)

OUT_PROACTIVE = Path(
    "routing/benchmarks/egowearbench/"
    "proactive_state_manifest.csv"
)


def load_jsonl(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                yield json.loads(line)


question_rows = []
proactive_rows = []


# ============================================================
# 1. EgoLongQA
# ============================================================

longqa_files = sorted(
    (ROOT / "egolongqa").glob("*.jsonl")
)

for path in longqa_files:

    for sample_idx, r in enumerate(
        load_jsonl(path)
    ):

        question_rows.append({
            "source": "EgoWearBench",
            "task": "egolongqa",
            "sample_id":
                f"egolongqa_{sample_idx}",

            "question":
                r["question"],

            "ground_truth":
                r["answer"],

            "routing_type":
                "TEMPORAL",

            "subtype":
                "long_video_qa",

            "category":
                r.get("category"),

            "video_path":
                r.get("video_path"),

            "interval_start": None,
            "interval_end": None,
        })


# ============================================================
# 2. EgoConv
#
# Flatten each question in questions[].
# Each question is aligned with answers[] and usually
# video_intervals[].
# ============================================================

conv_files = sorted(
    (ROOT / "egoconv").glob("*.jsonl")
)

for path in conv_files:

    for sample_idx, r in enumerate(
        load_jsonl(path)
    ):

        questions = (
            r.get("questions")
            or []
        )

        answers = (
            r.get("answers")
            or []
        )

        intervals = (
            r.get("video_intervals")
            or []
        )

        for turn_idx, q in enumerate(
            questions
        ):

            answer = (
                answers[turn_idx]
                if turn_idx < len(answers)
                else None
            )

            interval_start = None
            interval_end = None

            if turn_idx < len(intervals):
                interval = intervals[
                    turn_idx
                ]

                if (
                    isinstance(interval, list)
                    and len(interval) >= 2
                ):
                    interval_start = (
                        interval[0]
                    )
                    interval_end = (
                        interval[1]
                    )

            question_rows.append({
                "source":
                    "EgoWearBench",

                "task":
                    "egoconv",

                "sample_id":
                    (
                        f"egoconv_"
                        f"{sample_idx}_"
                        f"{turn_idx}"
                    ),

                "question":
                    q,

                "ground_truth":
                    answer,

                "routing_type":
                    "TEMPORAL",

                "subtype":
                    "conversational_video_qa",

                "category":
                    r.get("task"),

                "video_path":
                    r.get("video_path"),

                "interval_start":
                    interval_start,

                "interval_end":
                    interval_end,
            })


# ============================================================
# 3. EgoProactive
#
# A) Keep the initial user goal as a high-level
#    PROACTIVE query example.
#
# B) Expand every video/dialog state into a separate
#    WHEN training example.
# ============================================================

pro_files = sorted(
    (ROOT / "egoproactive").glob("*.jsonl")
)

for path in pro_files:

    for sample_idx, r in enumerate(
        load_jsonl(path)
    ):

        query = r.get("query")

        # ------------------------------------------
        # A. High-level routing example
        # ------------------------------------------

        if query:

            question_rows.append({
                "source":
                    "EgoWearBench",

                "task":
                    "egoproactive",

                "sample_id":
                    f"egoproactive_{sample_idx}",

                "question":
                    query,

                "ground_truth":
                    None,

                "routing_type":
                    "PROACTIVE",

                "subtype":
                    "goal_guidance",

                "category":
                    r.get("domain"),

                "video_path":
                    r.get("video_path"),

                "interval_start":
                    None,

                "interval_end":
                    None,
            })


        # ------------------------------------------
        # B. WHEN state-level examples
        # ------------------------------------------

        answers = (
            r.get("answers")
            or []
        )

        intervals = (
            r.get("video_intervals")
            or []
        )

        dialogs = (
            r.get("dialog")
            or []
        )

        n = max(
            len(answers),
            len(intervals),
            len(dialogs),
        )

        for state_idx in range(n):

            answer = (
                answers[state_idx]
                if state_idx < len(answers)
                else None
            )

            dialog_state = (
                dialogs[state_idx]
                if state_idx < len(dialogs)
                else None
            )

            interval_start = None
            interval_end = None

            if state_idx < len(intervals):

                interval = intervals[
                    state_idx
                ]

                if (
                    isinstance(interval, list)
                    and len(interval) >= 2
                ):
                    interval_start = (
                        interval[0]
                    )
                    interval_end = (
                        interval[1]
                    )

            proactive_rows.append({
                "source":
                    "EgoWearBench",

                "sample_id":
                    (
                        f"egoproactive_"
                        f"{sample_idx}_"
                        f"{state_idx}"
                    ),

                "query":
                    query,

                "domain":
                    r.get("domain"),

                "task":
                    r.get("task"),

                "video_path":
                    r.get("video_path"),

                "duration_in_sec":
                    r.get(
                        "duration_in_sec"
                    ),

                "state_index":
                    state_idx,

                "interval_start":
                    interval_start,

                "interval_end":
                    interval_end,

                "target_response":
                    answer,

                # Store dialog/context as JSON text
                "dialog_state_json":
                    json.dumps(
                        dialog_state,
                        ensure_ascii=False,
                    )
                    if dialog_state is not None
                    else None,
            })


# ============================================================
# Save
# ============================================================

questions = pd.DataFrame(
    question_rows
)

proactive = pd.DataFrame(
    proactive_rows
)


questions.to_csv(
    OUT_QUESTIONS,
    index=False,
)

proactive.to_csv(
    OUT_PROACTIVE,
    index=False,
)


# ============================================================
# Summary
# ============================================================

print("=" * 90)
print("EGOWEARBENCH QUESTION MANIFEST")
print("=" * 90)

print(
    "Total question rows:",
    len(questions),
)

print()
print("By task:")
print(
    questions["task"]
    .value_counts()
    .to_string()
)

print()
print("By routing type:")
print(
    questions["routing_type"]
    .value_counts()
    .to_string()
)

print()
print("By subtype:")
print(
    questions["subtype"]
    .value_counts()
    .to_string()
)


print()
print("=" * 90)
print("EGOPROACTIVE WHEN MANIFEST")
print("=" * 90)

print(
    "State-level examples:",
    len(proactive),
)

print()
print(
    "Unique videos:",
    proactive["video_path"]
    .nunique(),
)

print()
print(
    "Mean states/video:",
    len(proactive)
    / proactive["video_path"]
      .nunique(),
)

print()
print("Question manifest:")
print(OUT_QUESTIONS)

print()
print("WHEN manifest:")
print(OUT_PROACTIVE)
