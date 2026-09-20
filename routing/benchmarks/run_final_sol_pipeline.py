from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path.cwd()
BENCH = ROOT / "routing" / "benchmarks"

OUT = BENCH / "final_sol_labels"
OUT.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL = os.environ.get(
    "FINAL_LABEL_MODEL",
    "gpt-5.6-sol",
)

REASONING_EFFORT = os.environ.get(
    "FINAL_LABEL_REASONING",
    "medium",
)

TEXT_WORKERS = int(
    os.environ.get(
        "FINAL_TEXT_WORKERS",
        "6",
    )
)

IMAGE_WORKERS = int(
    os.environ.get(
        "FINAL_IMAGE_WORKERS",
        "4",
    )
)

# Wait up to 10 hours for the SuperGlasses phone benchmark.
SUPER_WAIT_MIN = int(
    os.environ.get(
        "SUPER_WAIT_MIN",
        "600",
    )
)

# Current GPT-5.6 Sol pricing.
# Used only for an approximate usage report.
INPUT_USD_PER_M = 4.0
OUTPUT_USD_PER_M = 20.0


if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY is not set."
    )


client = OpenAI(
    timeout=180,
    max_retries=5,
)


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def clean_scalar(x):
    if x is None:
        return None

    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    x = str(x).strip()

    if not x:
        return None

    if x.lower() in {
        "nan",
        "none",
        "null",
    }:
        return None

    return x


def first_value(row, names):
    for name in names:

        if name not in row:
            continue

        value = clean_scalar(
            row[name]
        )

        if value is not None:
            return value

    return None


def normalize_question(q):
    q = str(q).lower().strip()

    q = re.sub(
        r"\s+",
        " ",
        q,
    )

    return q


def image_to_data_url(path):
    path = Path(path)

    mime, _ = mimetypes.guess_type(
        str(path)
    )

    if not mime:
        mime = "image/jpeg"

    with path.open("rb") as f:
        b64 = base64.b64encode(
            f.read()
        ).decode("utf-8")

    return (
        f"data:{mime};base64,{b64}"
    )


def get_usage(response):
    usage = getattr(
        response,
        "usage",
        None,
    )

    if usage is None:
        return 0, 0, 0.0

    inp = int(
        getattr(
            usage,
            "input_tokens",
            0,
        )
        or 0
    )

    out = int(
        getattr(
            usage,
            "output_tokens",
            0,
        )
        or 0
    )

    cost = (
        inp
        * INPUT_USD_PER_M
        / 1_000_000
        +
        out
        * OUTPUT_USD_PER_M
        / 1_000_000
    )

    return inp, out, cost


def call_structured(
    prompt,
    schema_name,
    schema,
    image_path=None,
    effort=None,
):
    effort = (
        effort
        or REASONING_EFFORT
    )

    content = [
        {
            "type": "input_text",
            "text": prompt,
        }
    ]

    if image_path is not None:

        content.append({
            "type": "input_image",
            "image_url":
                image_to_data_url(
                    image_path
                ),
            "detail": "auto",
        })

    last_error = None

    for attempt in range(6):

        try:

            response = (
                client.responses.create(
                    model=MODEL,
                    reasoning={
                        "effort": effort,
                    },
                    input=[
                        {
                            "role": "user",
                            "content":
                                content,
                        }
                    ],
                    text={
                        "format": {
                            "type":
                                "json_schema",

                            "name":
                                schema_name,

                            "strict":
                                True,

                            "schema":
                                schema,
                        }
                    },
                    max_output_tokens=1200,
                )
            )

            parsed = json.loads(
                response.output_text
            )

            inp, out, cost = (
                get_usage(
                    response
                )
            )

            return {
                "parsed": parsed,
                "input_tokens": inp,
                "output_tokens": out,
                "estimated_cost_usd":
                    cost,
            }

        except Exception as e:

            last_error = e

            wait = min(
                2 ** attempt,
                30,
            )

            time.sleep(wait)

    raise RuntimeError(
        f"API failed after retries: "
        f"{type(last_error).__name__}: "
        f"{last_error}"
    )


def read_jsonl_latest(
    path,
    key="sample_id",
):
    path = Path(path)

    latest = {}

    if not path.exists():
        return pd.DataFrame()

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                r = json.loads(line)
            except Exception:
                continue

            sid = str(
                r.get(key)
            )

            latest[sid] = r

    return pd.DataFrame(
        latest.values()
    )


def completed_ids(
    path,
    key="sample_id",
):
    df = read_jsonl_latest(
        path,
        key=key,
    )

    if len(df) == 0:
        return set()

    good = df[
        df.get(
            "status",
            pd.Series(
                index=df.index,
                dtype=object,
            )
        )
        == "ok"
    ]

    if key not in good:
        return set()

    return set(
        good[key]
        .astype(str)
    )


def run_parallel(
    items,
    output_jsonl,
    worker,
    workers,
    description,
):
    output_jsonl = Path(
        output_jsonl
    )

    done = completed_ids(
        output_jsonl
    )

    pending = [
        x
        for x in items
        if str(
            x["sample_id"]
        )
        not in done
    ]

    print()
    print(
        description,
    )
    print(
        "Total    :",
        len(items),
    )
    print(
        "Completed:",
        len(done),
    )
    print(
        "Remaining:",
        len(pending),
    )

    if not pending:
        return

    with output_jsonl.open(
        "a",
        encoding="utf-8",
    ) as fout:

        with ThreadPoolExecutor(
            max_workers=workers
        ) as executor:

            futures = {
                executor.submit(
                    worker,
                    item,
                ): item
                for item in pending
            }

            for future in tqdm(
                as_completed(
                    futures
                ),
                total=len(futures),
                desc=description,
            ):

                item = futures[
                    future
                ]

                try:
                    result = (
                        future.result()
                    )

                    result[
                        "status"
                    ] = "ok"

                except Exception as e:

                    result = {
                        "sample_id":
                            item[
                                "sample_id"
                            ],

                        "status":
                            "error",

                        "error":
                            (
                                f"{type(e).__name__}: "
                                f"{e}"
                            ),
                    }

                fout.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                fout.flush()


# =============================================================================
# SCHEMAS
# =============================================================================

QUESTION_SCHEMA = {
    "type": "object",

    "properties": {
        "label": {
            "type": "string",
            "enum": [
                "DIRECT_VISUAL",
                "KNOWLEDGE",
                "TEMPORAL",
                "MEMORY",
            ],
        },

        "confidence": {
            "type": "number",
        },

        "reason": {
            "type": "string",
        },
    },

    "required": [
        "label",
        "confidence",
        "reason",
    ],

    "additionalProperties":
        False,
}


GRADE_SCHEMA = {
    "type": "object",

    "properties": {
        "label": {
            "type": "string",
            "enum": [
                "CORRECT",
                "PARTIAL",
                "INCORRECT",
                "AMBIGUOUS",
            ],
        },

        "confidence": {
            "type": "number",
        },

        "reason": {
            "type": "string",
        },
    },

    "required": [
        "label",
        "confidence",
        "reason",
    ],

    "additionalProperties":
        False,
}


# =============================================================================
# PROMPTS
# =============================================================================

def question_type_prompt(question):
    return f"""
You are creating FINAL training labels for an AI-glasses routing system.

Classify the QUESTION based on what information is fundamentally required
to answer it.

Use exactly one label.

DIRECT_VISUAL
- A current image or one relevant frame is normally enough.
- Includes object recognition, attributes, color, counting, OCR,
  spatial relations, current scene understanding, visible math,
  identifying visible objects, and current visual state.
- Do not call something TEMPORAL simply because it came from a video dataset.

KNOWLEDGE
- Requires external or general factual information that is not directly
  visible in the current scene.
- Examples include history, company information, factual background,
  material facts, dates, web/search information, or knowledge about an
  identified object/place.
- If the answer can simply be read or recognized from the image,
  use DIRECT_VISUAL instead.

TEMPORAL
- Requires information across multiple moments or frames.
- Includes before/after, event order, action progression, what happened,
  how something changed, next event based on observed sequence,
  or another genuinely multi-frame question.
- A question such as "what is this?" or "what color is this?" remains
  DIRECT_VISUAL even if it appears in a video benchmark.

MEMORY
- Requires recalling a user's earlier personal experience, previous
  conversation, earlier location, earlier action, or information from
  long-term episodic history.
- Examples: "Where did I leave my keys?" or
  "What did Alex tell me earlier?"

Important:
Judge the semantic requirement of the question itself.
Do NOT infer the label from the dataset source.

Question:
{question}

Return a concise reason.
""".strip()


def grade_prompt(
    question,
    reference,
    e2b_answer,
):
    return f"""
You are creating FINAL evaluation labels for a wearable AI glasses system.

You are given:
1. the actual image,
2. the user's question,
3. a reference answer,
4. the answer produced by the SMALL local vision-language model.

Judge whether the SMALL model's answer would be useful and correct for
the user.

Labels:

CORRECT
- The answer is substantively correct.
- It does not need to use exactly the same words as the reference.

PARTIAL
- The answer is substantially useful and close enough,
  but has a minor specificity, wording, or perceptual difference.
- The main answer is still useful.
- Examples can include a close visual category or minor color distinction
  when the image itself is somewhat ambiguous.

INCORRECT
- The answer is materially wrong, misleading, misses the essential answer,
  hallucinates important information, or fails to answer the question.

AMBIGUOUS
- The image/reference/question does not provide enough information to
  fairly determine correctness, or multiple interpretations are genuinely
  plausible.

Important:
- Use the IMAGE as evidence.
- Do not blindly assume the reference answer is perfect.
- Do not be overly strict about wording.
- A more detailed answer is fine if its important claims are correct.
- A partially useful answer should be PARTIAL, not automatically INCORRECT.

Question:
{question}

Reference answer:
{reference}

SMALL/E2B answer:
{e2b_answer}

Return a short reason.
""".strip()


# =============================================================================
# STAGE 1 — FINAL SOL EGOCONV LABELS
# =============================================================================

def run_egoconv():

    print()
    print("=" * 100)
    print("STAGE 1: SOL EGOCONV QUESTION-TYPE LABELS")
    print("=" * 100)

    src = (
        BENCH
        / "egowearbench"
        / "question_manifest.csv"
    )

    df = pd.read_csv(
        src
    )

    df = df[
        df["task"]
        == "egoconv"
    ].copy()

    items = []

    for _, r in df.iterrows():

        items.append({
            "sample_id":
                str(
                    r["sample_id"]
                ),

            "question":
                str(
                    r["question"]
                ),
        })

    jsonl = (
        OUT
        / "egoconv_sol.jsonl"
    )

    def worker(item):

        result = call_structured(
            question_type_prompt(
                item["question"]
            ),
            "question_type_label",
            QUESTION_SCHEMA,
        )

        p = result["parsed"]

        return {
            "sample_id":
                item["sample_id"],

            "question":
                item["question"],

            "sol_label":
                p["label"],

            "sol_confidence":
                p["confidence"],

            "sol_reason":
                p["reason"],

            "input_tokens":
                result[
                    "input_tokens"
                ],

            "output_tokens":
                result[
                    "output_tokens"
                ],

            "estimated_cost_usd":
                result[
                    "estimated_cost_usd"
                ],
        }

    run_parallel(
        items,
        jsonl,
        worker,
        TEXT_WORKERS,
        "EgoConv Sol labeling",
    )

    sol = read_jsonl_latest(
        jsonl
    )

    sol = sol[
        sol["status"] == "ok"
    ].copy()

    # ---------------------------------------------------------
    # Compare against prior Luna labels if available.
    # ---------------------------------------------------------

    luna_path = (
        BENCH
        / "egowearbench"
        / "egoconv_routing_labels.csv"
    )

    if luna_path.exists():

        luna = pd.read_csv(
            luna_path
        )

        if (
            "sample_id" in luna
            and
            "new_routing_type"
            in luna
        ):

            luna = luna[
                [
                    "sample_id",
                    "new_routing_type",
                ]
            ].copy()

            luna[
                "sample_id"
            ] = (
                luna["sample_id"]
                .astype(str)
            )

            luna = luna.rename(
                columns={
                    "new_routing_type":
                        "luna_label"
                }
            )

            sol[
                "sample_id"
            ] = (
                sol["sample_id"]
                .astype(str)
            )

            sol = sol.merge(
                luna,
                on="sample_id",
                how="left",
            )

            sol[
                "luna_sol_agree"
            ] = (
                sol["luna_label"]
                ==
                sol["sol_label"]
            )

            disagreements = sol[
                sol["luna_label"].notna()
                &
                (
                    sol["luna_label"]
                    !=
                    sol["sol_label"]
                )
            ]

            disagreements.to_csv(
                OUT
                / "egoconv_luna_sol_disagreements.csv",
                index=False,
            )

            print()
            print(
                "Luna/Sol agreement:",
                round(
                    sol[
                        "luna_sol_agree"
                    ].mean()
                    * 100,
                    2,
                ),
                "%",
            )

            print(
                "Disagreements:",
                len(
                    disagreements
                ),
            )

    out_csv = (
        OUT
        / "egoconv_question_type_final.csv"
    )

    sol.to_csv(
        out_csv,
        index=False,
    )

    print()
    print(
        sol["sol_label"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Saved:",
        out_csv,
    )

    return sol


# =============================================================================
# STAGE 2 — BUILD FINAL 4-WAY CORPUS
# =============================================================================

def build_final_corpus(
    egoconv_sol,
):

    print()
    print("=" * 100)
    print("STAGE 2: BUILD FINAL 4-WAY QUESTION CORPUS")
    print("=" * 100)

    rows = []

    # ---------------------------------------------------------
    # WearVQA
    # ---------------------------------------------------------

    wear = pd.read_csv(
        BENCH
        / "wearvqa_full"
        / "manifest.csv"
    )

    for i, r in wear.iterrows():

        rows.append({
            "source":
                "WearVQA",

            "sample_id":
                first_value(
                    r,
                    [
                        "sample_id",
                        "id",
                    ],
                )
                or
                f"wear_{i}",

            "question":
                first_value(
                    r,
                    ["question"],
                ),

            "routing_type":
                "DIRECT_VISUAL",

            "subtype":
                first_value(
                    r,
                    [
                        "category",
                        "question_type",
                    ],
                ),
        })

    # ---------------------------------------------------------
    # SuperGlasses
    # ---------------------------------------------------------

    sg = pd.read_csv(
        BENCH
        / "superglasses"
        / "full_routing_manifest.csv"
    )

    for i, r in sg.iterrows():

        label = first_value(
            r,
            ["routing_type"],
        )

        if label not in {
            "DIRECT_VISUAL",
            "KNOWLEDGE",
            "TEMPORAL",
            "MEMORY",
        }:
            continue

        rows.append({
            "source":
                "SuperGlasses",

            "sample_id":
                first_value(
                    r,
                    [
                        "sample_id",
                        "id",
                    ],
                )
                or
                f"sg_{i}",

            "question":
                first_value(
                    r,
                    ["question"],
                ),

            "routing_type":
                label,

            "subtype":
                first_value(
                    r,
                    [
                        "categories",
                        "category",
                    ],
                ),
        })

    # ---------------------------------------------------------
    # SuperMemory
    # ---------------------------------------------------------

    sm = pd.read_csv(
        BENCH
        / "supermemory"
        / "question_manifest.csv"
    )

    for i, r in sm.iterrows():

        rows.append({
            "source":
                "SuperMemory-VQA",

            "sample_id":
                first_value(
                    r,
                    ["sample_id"],
                )
                or
                f"sm_{i}",

            "question":
                first_value(
                    r,
                    ["question"],
                ),

            "routing_type":
                "MEMORY",

            "subtype":
                first_value(
                    r,
                    ["subtype"],
                ),
        })

    # ---------------------------------------------------------
    # EgoWearBench
    # ---------------------------------------------------------

    ego = pd.read_csv(
        BENCH
        / "egowearbench"
        / "question_manifest.csv"
    )

    label_map = dict(
        zip(
            egoconv_sol[
                "sample_id"
            ].astype(str),

            egoconv_sol[
                "sol_label"
            ],
        )
    )

    for i, r in ego.iterrows():

        task = first_value(
            r,
            ["task"],
        )

        # PROACTIVE belongs to WHEN, not WHICH.
        if task == "egoproactive":
            continue

        sid = first_value(
            r,
            ["sample_id"],
        )

        if task == "egoconv":

            label = label_map.get(
                str(sid)
            )

            if label is None:
                continue

        elif task == "egolongqa":

            label = "TEMPORAL"

        else:

            label = first_value(
                r,
                ["routing_type"],
            )

        if label not in {
            "DIRECT_VISUAL",
            "KNOWLEDGE",
            "TEMPORAL",
            "MEMORY",
        }:
            continue

        rows.append({
            "source":
                "EgoWearBench",

            "sample_id":
                sid
                or
                f"ego_{i}",

            "question":
                first_value(
                    r,
                    ["question"],
                ),

            "routing_type":
                label,

            "subtype":
                first_value(
                    r,
                    ["subtype"],
                ),
        })

    df = pd.DataFrame(
        rows
    )

    df = df[
        df["question"].notna()
    ].copy()

    df[
        "question_norm"
    ] = (
        df["question"]
        .astype(str)
        .map(
            normalize_question
        )
    )

    # ---------------------------------------------------------
    # Remove questions assigned conflicting labels.
    # ---------------------------------------------------------

    label_counts = (
        df.groupby(
            "question_norm"
        )["routing_type"]
        .nunique()
    )

    conflict_q = set(
        label_counts[
            label_counts > 1
        ].index
    )

    conflicts = df[
        df["question_norm"]
        .isin(
            conflict_q
        )
    ].copy()

    conflicts.to_csv(
        OUT
        / "final_question_type_conflicts.csv",
        index=False,
    )

    clean = df[
        ~df["question_norm"]
        .isin(
            conflict_q
        )
    ].copy()

    clean = (
        clean
        .drop_duplicates(
            subset=[
                "question_norm",
                "routing_type",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    out = (
        OUT
        / "final_question_type_corpus.csv"
    )

    clean.to_csv(
        out,
        index=False,
    )

    print(
        "Rows:",
        len(clean),
    )

    print(
        "Conflicting questions removed:",
        len(
            conflict_q
        ),
    )

    print()
    print(
        clean[
            "routing_type"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        pd.crosstab(
            clean["source"],
            clean[
                "routing_type"
            ],
        ).to_string()
    )

    print()
    print(
        "Saved:",
        out,
    )

    return clean


# =============================================================================
# STAGE 3 — TRAIN FINAL QUESTION-TYPE ROUTER
# =============================================================================

def train_question_router(
    df,
):

    print()
    print("=" * 100)
    print("STAGE 3: TRAIN QUESTION-TYPE ROUTER")
    print("=" * 100)

    X = (
        df["question"]
        .astype(str)
        .to_numpy(dtype=object)
    )

    y = (
        df["routing_type"]
        .astype(str)
        .to_numpy(dtype=object)
    )

    # 80 / 10 / 10
    X_trainval, X_test, y_trainval, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.10,
            random_state=42,
            stratify=y,
        )
    )

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_trainval,
            y_trainval,
            test_size=1 / 9,
            random_state=42,
            stratify=y_trainval,
        )
    )

    best_c = None
    best_f1 = -1

    for C in [
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
    ]:

        pipe = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(
                        1,
                        2,
                    ),
                    min_df=2,
                    max_features=70000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=4000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ])

        pipe.fit(
            X_train,
            y_train,
        )

        pred = pipe.predict(
            X_val
        )

        f1 = f1_score(
            y_val,
            pred,
            average="macro",
        )

        print(
            f"C={C:<4} "
            f"val macro-F1="
            f"{f1:.4f}"
        )

        if f1 > best_f1:

            best_f1 = f1
            best_c = C

    print()
    print(
        "Selected C:",
        best_c,
    )

    final_model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(
                    1,
                    2,
                ),
                min_df=2,
                max_features=70000,
                sublinear_tf=True,
                strip_accents="unicode",
            ),
        ),
        (
            "clf",
            LogisticRegression(
                C=best_c,
                max_iter=4000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    final_model.fit(
        X_trainval,
        y_trainval,
    )

    test_pred = (
        final_model.predict(
            X_test
        )
    )

    acc = accuracy_score(
        y_test,
        test_pred,
    )

    macro = f1_score(
        y_test,
        test_pred,
        average="macro",
    )

    print()
    print(
        "TEST ACCURACY :",
        round(
            acc,
            4,
        ),
    )

    print(
        "TEST MACRO-F1 :",
        round(
            macro,
            4,
        ),
    )

    print()
    print(
        classification_report(
            y_test,
            test_pred,
            digits=4,
        )
    )

    labels = [
        "DIRECT_VISUAL",
        "KNOWLEDGE",
        "TEMPORAL",
        "MEMORY",
    ]

    cm = confusion_matrix(
        y_test,
        test_pred,
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )

    print(
        "CONFUSION MATRIX"
    )

    print(
        cm_df.to_string()
    )

    cm_df.to_csv(
        OUT
        / "question_type_confusion_matrix.csv"
    )

    model_path = (
        ROOT
        / "routing"
        / "question_type_router_sol.joblib"
    )

    joblib.dump(
        {
            "model":
                final_model,

            "classes":
                labels,

            "best_C":
                best_c,

            "validation_macro_f1":
                best_f1,

            "test_accuracy":
                acc,

            "test_macro_f1":
                macro,

            "training_source":
                str(
                    OUT
                    / "final_question_type_corpus.csv"
                ),

            "label_model":
                MODEL,
        },
        model_path,
    )

    print()
    print(
        "Saved model:",
        model_path,
    )


# =============================================================================
# IMAGE RESOLUTION HELPERS
# =============================================================================

def build_image_index(
    image_dir,
):
    image_dir = Path(
        image_dir
    )

    by_name = {}
    by_stem = {}

    if not image_dir.exists():
        return by_name, by_stem

    valid = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    for p in image_dir.rglob("*"):

        if (
            p.is_file()
            and
            p.suffix.lower()
            in valid
        ):

            by_name.setdefault(
                p.name,
                p,
            )

            by_stem.setdefault(
                p.stem,
                p,
            )

    return (
        by_name,
        by_stem,
    )


def resolve_image(
    row,
    image_dir,
    by_name,
    by_stem,
):

    for col in [
        "image_path",
        "local_image_path",
        "filename",
        "file_name",
        "image_file",
        "image_path_manifest",
    ]:

        if col not in row:
            continue

        value = clean_scalar(
            row[col]
        )

        if not value:
            continue

        p = Path(value)

        for candidate in [
            p,
            ROOT / p,
            Path(image_dir)
            / p.name,
        ]:

            if candidate.exists():
                return candidate

        if p.name in by_name:
            return by_name[
                p.name
            ]

        if p.stem in by_stem:
            return by_stem[
                p.stem
            ]

    for col in [
        "image_id",
        "sample_id",
        "id",
        "source_id",
        "souce_id",
    ]:

        if col not in row:
            continue

        value = clean_scalar(
            row[col]
        )

        if not value:
            continue

        if value in by_name:
            return by_name[
                value
            ]

        stem = Path(
            value
        ).stem

        if stem in by_stem:
            return by_stem[
                stem
            ]

    return None


# =============================================================================
# GENERIC E2B GRADING
# =============================================================================

def grade_e2b_dataframe(
    df,
    dataset_name,
    image_dir,
    output_prefix,
):

    by_name, by_stem = (
        build_image_index(
            image_dir
        )
    )

    print()
    print(
        dataset_name,
        "images indexed:",
        len(by_name),
    )

    items = []

    skipped = 0

    for i, r in df.iterrows():

        sid = (
            first_value(
                r,
                [
                    "sample_id",
                    "id",
                    "image_id",
                ],
            )
            or
            f"{dataset_name}_{i}"
        )

        question = (
            first_value(
                r,
                [
                    "question",
                    "question_manifest",
                ],
            )
        )

        reference = (
            first_value(
                r,
                [
                    "reference_answer",
                    "ground_truth",
                    "gt_answer",
                    "answer",
                    "response",
                    "ground_truth_manifest",
                    "answer_manifest",
                    "response_manifest",
                ],
            )
        )

        e2b = (
            first_value(
                r,
                [
                    "e2b_answer",
                    "model_answer",
                    "prediction",
                    "e2b_response",
                    "small_answer",
                    "predicted_answer",
                ],
            )
        )

        image = resolve_image(
            r,
            image_dir,
            by_name,
            by_stem,
        )

        if (
            not question
            or
            not reference
            or
            not e2b
            or
            image is None
        ):
            skipped += 1
            continue

        old_label = None

        for col in [
            "judge_label",
            "image_judge_label",
            "grade",
            "semantic_label",
        ]:

            if col not in r:
                continue

            x = clean_scalar(
                r[col]
            )

            if x and x.upper() in {
                "CORRECT",
                "PARTIAL",
                "INCORRECT",
                "AMBIGUOUS",
            }:

                old_label = (
                    x.upper()
                )

                break

        items.append({
            "sample_id":
                str(sid),

            "question":
                question,

            "reference_answer":
                reference,

            "e2b_answer":
                e2b,

            "image_path":
                str(image),

            "old_luna_label":
                old_label,
        })

    print(
        "Gradeable:",
        len(items),
    )

    print(
        "Skipped  :",
        skipped,
    )

    jsonl = (
        OUT
        / f"{output_prefix}.jsonl"
    )

    def worker(item):

        result = call_structured(
            grade_prompt(
                item["question"],
                item[
                    "reference_answer"
                ],
                item[
                    "e2b_answer"
                ],
            ),
            "answer_grade",
            GRADE_SCHEMA,
            image_path=
                item["image_path"],
        )

        p = result[
            "parsed"
        ]

        return {
            **item,

            "sol_label":
                p["label"],

            "sol_confidence":
                p["confidence"],

            "sol_reason":
                p["reason"],

            "input_tokens":
                result[
                    "input_tokens"
                ],

            "output_tokens":
                result[
                    "output_tokens"
                ],

            "estimated_cost_usd":
                result[
                    "estimated_cost_usd"
                ],
        }

    run_parallel(
        items,
        jsonl,
        worker,
        IMAGE_WORKERS,
        f"{dataset_name} Sol grading",
    )

    result = (
        read_jsonl_latest(
            jsonl
        )
    )

    result = result[
        result["status"]
        == "ok"
    ].copy()

    # Router binary label.
    result[
        "small_sufficient"
    ] = result[
        "sol_label"
    ].map({
        "CORRECT": 1,
        "PARTIAL": 1,
        "INCORRECT": 0,
        "AMBIGUOUS": pd.NA,
    })

    # Luna/Sol comparison where available.
    if (
        "old_luna_label"
        in result
    ):

        known = result[
            result[
                "old_luna_label"
            ].notna()
        ].copy()

        if len(known):

            known[
                "luna_sol_agree"
            ] = (
                known[
                    "old_luna_label"
                ]
                ==
                known[
                    "sol_label"
                ]
            )

            disagree = known[
                ~known[
                    "luna_sol_agree"
                ]
            ]

            disagree.to_csv(
                OUT
                / (
                    f"{output_prefix}"
                    "_luna_sol_disagreements.csv"
                ),
                index=False,
            )

            print(
                "Luna/Sol agreement:",
                round(
                    known[
                        "luna_sol_agree"
                    ].mean()
                    * 100,
                    2,
                ),
                "%",
            )

            print(
                "Disagreements:",
                len(
                    disagree
                ),
            )

    csv = (
        OUT
        / f"{output_prefix}.csv"
    )

    result.to_csv(
        csv,
        index=False,
    )

    print()
    print(
        result[
            "sol_label"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    judgeable = result[
        result[
            "small_sufficient"
        ].notna()
    ]

    if len(judgeable):

        print()
        print(
            "SMALL sufficient:",
            round(
                judgeable[
                    "small_sufficient"
                ]
                .astype(float)
                .mean()
                * 100,
                2,
            ),
            "%",
        )

        print(
            "SMALL failure   :",
            round(
                (
                    1
                    -
                    judgeable[
                        "small_sufficient"
                    ]
                    .astype(float)
                    .mean()
                )
                * 100,
                2,
            ),
            "%",
        )

    print()
    print(
        "Saved:",
        csv,
    )

    return result


# =============================================================================
# STAGE 4 — WEARVQA SOL RE-GRADING
# =============================================================================

def load_wearvqa():

    old_path = (
        BENCH
        / "results"
        / "wearvqa_e2b_image_graded.csv"
    )

    manifest_path = (
        BENCH
        / "wearvqa_full"
        / "manifest.csv"
    )

    if not old_path.exists():

        raise FileNotFoundError(
            f"Cannot find existing "
            f"WearVQA grading file: "
            f"{old_path}"
        )

    old = pd.read_csv(
        old_path
    )

    manifest = pd.read_csv(
        manifest_path
    )

    if (
        "sample_id" in old
        and
        "sample_id" in manifest
    ):

        old[
            "sample_id"
        ] = (
            old["sample_id"]
            .astype(str)
        )

        manifest[
            "sample_id"
        ] = (
            manifest["sample_id"]
            .astype(str)
        )

        merged = old.merge(
            manifest,
            on="sample_id",
            how="left",
            suffixes=(
                "",
                "_manifest",
            ),
        )

        return merged

    return old


def run_wearvqa_grade():

    print()
    print("=" * 100)
    print("STAGE 4: SOL RE-GRADE WEARVQA E2B")
    print("=" * 100)

    df = load_wearvqa()

    print(
        "Rows:",
        len(df),
    )

    print(
        "Columns:",
        list(df.columns),
    )

    return grade_e2b_dataframe(
        df,
        "WearVQA",
        BENCH
        / "wearvqa_full"
        / "images",
        "wearvqa_e2b_sol_graded",
    )


# =============================================================================
# STAGE 5 — WAIT FOR SUPERGLASSES E2B
# =============================================================================

def load_superglasses_e2b():

    result_dir = (
        BENCH
        / "superglasses"
        / "results"
    )

    csv_path = (
        result_dir
        / "superglasses_direct_visual_e2b.csv"
    )

    jsonl_path = (
        result_dir
        / "superglasses_direct_visual_e2b.jsonl"
    )

    manifest_path = (
        BENCH
        / "superglasses"
        / "direct_visual_manifest.csv"
    )

    expected = len(
        pd.read_csv(
            manifest_path
        )
    )

    print()
    print(
        "Expected SuperGlasses:",
        expected,
    )

    start = time.time()

    while True:

        if csv_path.exists():

            try:
                df = pd.read_csv(
                    csv_path
                )

                unique = (
                    df["sample_id"]
                    .astype(str)
                    .nunique()
                    if "sample_id" in df
                    else len(df)
                )

                print(
                    "SuperGlasses E2B rows:",
                    unique,
                    "/",
                    expected,
                )

                if unique >= expected:
                    return df

            except Exception:
                pass

        elapsed_min = (
            time.time()
            - start
        ) / 60

        if (
            elapsed_min
            >= SUPER_WAIT_MIN
        ):

            print(
                "Wait timeout reached."
            )

            break

        print(
            "SuperGlasses E2B not finished yet. "
            "Checking again in 60 seconds..."
        )

        time.sleep(60)

    # Fall back to whatever exists.
    if csv_path.exists():

        return pd.read_csv(
            csv_path
        )

    if jsonl_path.exists():

        rows = []

        with jsonl_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                try:
                    rows.append(
                        json.loads(
                            line
                        )
                    )
                except Exception:
                    pass

        df = pd.DataFrame(
            rows
        )

        if (
            len(df)
            and
            "sample_id" in df
        ):

            df = (
                df
                .drop_duplicates(
                    subset=[
                        "sample_id"
                    ],
                    keep="last",
                )
            )

        return df

    print(
        "No SuperGlasses E2B "
        "result found."
    )

    return None


# =============================================================================
# STAGE 6 — SUPERGLASSES SOL GRADING
# =============================================================================

def run_superglasses_grade():

    print()
    print("=" * 100)
    print("STAGE 5/6: WAIT + SOL GRADE SUPERGLASSES")
    print("=" * 100)

    df = (
        load_superglasses_e2b()
    )

    if df is None:
        return None

    # Remove failed phone inferences.
    if "error" in df:

        df = df[
            df["error"].isna()
        ].copy()

    print(
        "Available successful rows:",
        len(df),
    )

    return grade_e2b_dataframe(
        df,
        "SuperGlasses",
        BENCH
        / "superglasses"
        / "images",
        "superglasses_e2b_sol_graded",
    )


# =============================================================================
# FINAL SUMMARY
# =============================================================================

def print_cost_summary():

    print()
    print("=" * 100)
    print("FINAL SOL LABELING SUMMARY")
    print("=" * 100)

    total_input = 0
    total_output = 0
    total_cost = 0.0

    for path in OUT.glob(
        "*.jsonl"
    ):

        df = read_jsonl_latest(
            path
        )

        if not len(df):
            continue

        inp = pd.to_numeric(
            df.get(
                "input_tokens",
                0,
            ),
            errors="coerce",
        ).fillna(0).sum()

        out = pd.to_numeric(
            df.get(
                "output_tokens",
                0,
            ),
            errors="coerce",
        ).fillna(0).sum()

        cost = pd.to_numeric(
            df.get(
                "estimated_cost_usd",
                0,
            ),
            errors="coerce",
        ).fillna(0).sum()

        total_input += inp
        total_output += out
        total_cost += cost

    print(
        "Model:",
        MODEL,
    )

    print(
        "Reasoning:",
        REASONING_EFFORT,
    )

    print(
        "Input tokens :",
        int(total_input),
    )

    print(
        "Output tokens:",
        int(total_output),
    )

    print(
        "Approx API cost: $",
        round(
            total_cost,
            2,
        ),
    )

    print()
    print(
        "Final output directory:"
    )

    print(
        OUT
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print("FINAL AI-GLASSES SOL LABELING PIPELINE")
    print("=" * 100)

    print(
        "Model:",
        MODEL,
    )

    print(
        "Reasoning:",
        REASONING_EFFORT,
    )

    print(
        "Text workers:",
        TEXT_WORKERS,
    )

    print(
        "Image workers:",
        IMAGE_WORKERS,
    )

    # 1. Correct EgoConv labels.
    egoconv = run_egoconv()

    # 2. Rebuild final question corpus.
    corpus = build_final_corpus(
        egoconv
    )

    # 3. Train question-type classifier.
    train_question_router(
        corpus
    )

    # 4. Rejudge old 2500 WearVQA.
    run_wearvqa_grade()

    # 5/6. Wait for ~977 SuperGlasses
    # phone runs and grade them.
    run_superglasses_grade()

    # Final cost.
    print_cost_summary()

    print()
    print("=" * 100)
    print("ALL DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
