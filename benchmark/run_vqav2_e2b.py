import argparse
import json
import re
import string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


# ============================================================
# Existing Galaxy E2B bridge
# ============================================================

BRIDGE_DIR = Path.home() / "projectaria_client_sdk_samples"
sys.path.insert(0, str(BRIDGE_DIR))

from phone_bridge import ask_phone


TMP_IMAGE = Path("/tmp/vqav2_e2b_input.jpg")


# ============================================================
# VQAv2 answer normalization
# Based on standard VQA evaluation behavior
# ============================================================

CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldnt": "couldn't",
    "couldve": "could've",
    "couldntve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "id": "i'd",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itll": "it'll",
    "lets": "let's",
    "mightnt": "mightn't",
    "mustnt": "mustn't",
    "shant": "shan't",
    "shed": "she'd",
    "shell": "she'll",
    "shes": "she's",
    "shouldnt": "shouldn't",
    "shouldve": "should've",
    "thats": "that's",
    "theres": "there's",
    "theyd": "they'd",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "wed": "we'd",
    "well": "we'll",
    "were": "we're",
    "werent": "weren't",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whats": "what's",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whod": "who'd",
    "wholl": "who'll",
    "whos": "who's",
    "whyd": "why'd",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

NUMBER_WORDS = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

ARTICLES = {"a", "an", "the"}

PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
PUNCT = [
    ";", "/", "[", "]", '"', "{", "}", "(", ")", "=",
    "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?",
    "!", ":"
]


def normalize_answer(text):
    text = str(text).replace("\n", " ").replace("\t", " ").strip().lower()

    for p in PUNCT:
        if (p + " " in text) or (" " + p in text) or re.search(COMMA_STRIP, text):
            text = text.replace(p, "")
        else:
            text = text.replace(p, " ")

    text = PERIOD_STRIP.sub("", text)

    words = []
    for word in text.split():
        word = NUMBER_WORDS.get(word, word)

        if word in ARTICLES:
            continue

        word = CONTRACTIONS.get(word, word)
        words.append(word)

    return " ".join(words)


# ============================================================
# Bridge result parsing
# ============================================================

def extract_answer(result):
    if result is None:
        return ""

    if isinstance(result, tuple):
        if len(result) == 0:
            return ""
        return str(result[0]).strip()

    if isinstance(result, dict):
        for key in ("answer", "text", "response", "result", "output"):
            if key in result:
                return str(result[key]).strip()
        return str(result).strip()

    return str(result).strip()


def clean_model_answer(answer):
    """
    E2B is explicitly asked for a short VQA answer.
    Remove a few harmless wrappers if they appear anyway.
    """
    answer = str(answer).strip()

    prefixes = [
        "the answer is ",
        "answer: ",
        "the answer is: ",
        "it is ",
        "it's ",
    ]

    low = answer.lower()

    for prefix in prefixes:
        if low.startswith(prefix):
            answer = answer[len(prefix):].strip()
            break

    # Remove surrounding quotes.
    answer = answer.strip(" \"'")

    # Short VQA prompts should not produce paragraphs.
    # If it still does, keep the first line.
    if "\n" in answer:
        answer = answer.split("\n", 1)[0].strip()

    return answer


# ============================================================
# Ground truth extraction
# ============================================================

def get_human_answers(example):
    raw = example["answers"]
    output = []

    for item in raw:
        if isinstance(item, dict):
            output.append(str(item.get("answer", "")))
        else:
            output.append(str(item))

    return output


# ============================================================
# Official-style VQA consensus score
#
# VQA evaluates prediction against 10 human answers.
# For each annotator, compare prediction with the other 9,
# then average:
#
# min(matches / 3, 1)
#
# This produces the familiar 0.0 / 0.3 / 0.6 / 0.9 / 1.0
# behavior for increasing human agreement.
# ============================================================

def vqa_consensus_score(prediction, human_answers):
    pred = normalize_answer(prediction)
    gt = [normalize_answer(x) for x in human_answers]

    if not pred or not gt:
        return 0.0

    per_annotator = []

    for i in range(len(gt)):
        other_answers = gt[:i] + gt[i + 1:]
        matches = sum(1 for x in other_answers if x == pred)
        per_annotator.append(min(1.0, matches / 3.0))

    return sum(per_annotator) / len(per_annotator)


# ============================================================
# Existing-result handling / resume
# ============================================================

def load_completed(path):
    completed = {}

    if not path.exists():
        return completed

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                completed[str(row["question_id"])] = row
            except Exception:
                continue

    return completed


# ============================================================
# Summary generation
# ============================================================

def create_summary(rows, output_path):
    df = pd.DataFrame(rows)

    if len(df) == 0:
        print("No results to summarize.")
        return

    valid = df[df["error"].isna() | (df["error"] == "")].copy()

    print()
    print("=" * 76)
    print("FINAL E2B VQAv2 BENCHMARK")
    print("=" * 76)

    print(f"Attempted cases       : {len(df):,}")
    print(f"Successful inference  : {len(valid):,}")
    print(f"Inference errors      : {len(df) - len(valid):,}")

    if len(valid) == 0:
        return

    accuracy = valid["vqa_score"].mean()
    strict_acc = valid["canonical_exact"].mean()
    latency_mean = valid["latency_s"].mean()
    latency_median = valid["latency_s"].median()
    latency_p95 = valid["latency_s"].quantile(0.95)

    print(f"VQA consensus accuracy: {accuracy * 100:.2f}%")
    print(f"Canonical exact acc   : {strict_acc * 100:.2f}%")
    print(f"Mean latency          : {latency_mean:.2f} s")
    print(f"Median latency        : {latency_median:.2f} s")
    print(f"P95 latency           : {latency_p95:.2f} s")

    # --------------------------------------------------------
    # Question type
    # --------------------------------------------------------

    qtype = (
        valid.groupby("question_type")
        .agg(
            n=("question_id", "count"),
            vqa_accuracy=("vqa_score", "mean"),
            canonical_exact_accuracy=("canonical_exact", "mean"),
            mean_latency_s=("latency_s", "mean"),
        )
        .reset_index()
        .sort_values(["n", "vqa_accuracy"], ascending=[False, False])
    )

    qtype["vqa_accuracy"] *= 100
    qtype["canonical_exact_accuracy"] *= 100

    qtype_path = output_path.with_name(
        output_path.stem + "_by_question_type.csv"
    )
    qtype.to_csv(qtype_path, index=False)

    # --------------------------------------------------------
    # Answer type
    # --------------------------------------------------------

    atype = (
        valid.groupby("answer_type")
        .agg(
            n=("question_id", "count"),
            vqa_accuracy=("vqa_score", "mean"),
            canonical_exact_accuracy=("canonical_exact", "mean"),
            mean_latency_s=("latency_s", "mean"),
        )
        .reset_index()
    )

    atype["vqa_accuracy"] *= 100
    atype["canonical_exact_accuracy"] *= 100

    atype_path = output_path.with_name(
        output_path.stem + "_by_answer_type.csv"
    )
    atype.to_csv(atype_path, index=False)

    # --------------------------------------------------------
    # Full CSV
    # --------------------------------------------------------

    full_csv = output_path.with_suffix(".csv")
    df.to_csv(full_csv, index=False)

    # --------------------------------------------------------
    # Wrong / low-score cases
    # --------------------------------------------------------

    wrong = valid[valid["vqa_score"] < 1.0].copy()
    wrong_path = output_path.with_name(
        output_path.stem + "_errors.csv"
    )
    wrong.to_csv(wrong_path, index=False)

    # --------------------------------------------------------
    # Human-readable summary text
    # --------------------------------------------------------

    summary_txt = output_path.with_name(
        output_path.stem + "_summary.txt"
    )

    with summary_txt.open("w", encoding="utf-8") as f:
        f.write("E2B VQAv2 BENCHMARK\n")
        f.write("=" * 70 + "\n")
        f.write(f"Attempted cases: {len(df):,}\n")
        f.write(f"Successful inference: {len(valid):,}\n")
        f.write(f"Inference errors: {len(df) - len(valid):,}\n")
        f.write(f"VQA consensus accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"Canonical exact accuracy: {strict_acc * 100:.2f}%\n")
        f.write(f"Mean latency: {latency_mean:.3f} s\n")
        f.write(f"Median latency: {latency_median:.3f} s\n")
        f.write(f"P95 latency: {latency_p95:.3f} s\n")

    print()
    print("Output files")
    print("-" * 76)
    print(f"Raw JSONL       : {output_path}")
    print(f"Full CSV        : {full_csv}")
    print(f"Question types  : {qtype_path}")
    print(f"Answer types    : {atype_path}")
    print(f"Error cases     : {wrong_path}")
    print(f"Summary         : {summary_txt}")

    print()
    print("Accuracy by answer type")
    print("-" * 76)

    for _, r in atype.iterrows():
        print(
            f"{r['answer_type']:<12} "
            f"n={int(r['n']):>6}  "
            f"acc={r['vqa_accuracy']:6.2f}%  "
            f"lat={r['mean_latency_s']:.2f}s"
        )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n",
        type=int,
        default=30000,
        help="Number of VQAv2 validation questions to benchmark.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="benchmark/results/vqav2_e2b_30k.jsonl",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("VQAv2 → Galaxy Gemma E2B benchmark")
    print("=" * 76)
    print(f"Target samples : {args.n:,}")
    print(f"Random seed    : {args.seed}")
    print(f"Output         : {output_path}")
    print()

    # --------------------------------------------------------
    # Load VQAv2
    # --------------------------------------------------------

    print("Loading VQAv2 validation dataset...")

    dataset = load_dataset(
        "parquet",
        data_files={
            "validation": "hf://datasets/lmms-lab/VQAv2/data/validation-*.parquet"
        },
        split="validation",
    )

    print(f"Available questions: {len(dataset):,}")

    n = min(args.n, len(dataset))

    # Fixed random sample, reproducible across resumed runs.
    dataset = dataset.shuffle(seed=args.seed).select(range(n))

    selected_ids = set(str(x) for x in dataset["question_id"])

    completed = load_completed(output_path)

    already_done = len(selected_ids.intersection(completed.keys()))

    print(f"Selected questions : {n:,}")
    print(f"Already completed  : {already_done:,}")
    print(f"Remaining          : {n - already_done:,}")
    print()

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    for ex in tqdm(dataset, total=n, desc="Galaxy E2B"):
        qid = str(ex["question_id"])

        if qid in completed:
            continue

        question = str(ex["question"])

        prompt = (
            f"{question}\n"
            "Answer with only the shortest possible answer. "
            "Use one word, one number, yes/no, or a short phrase. "
            "Do not explain and do not write a full sentence."
        )

        image = ex["image"].convert("RGB")
        image.save(TMP_IMAGE, format="JPEG", quality=95)

        answer = ""
        error = None
        latency = None

        # ----------------------------------------------------
        # Retry bridge failures
        # ----------------------------------------------------

        for attempt in range(args.retries + 1):
            t0 = time.perf_counter()

            try:
                result = ask_phone(
                    str(TMP_IMAGE),
                    prompt,
                    timeout=args.timeout,
                )

                latency = time.perf_counter() - t0

                answer = clean_model_answer(
                    extract_answer(result)
                )

                if answer:
                    error = None
                    break

                raise RuntimeError("Galaxy returned an empty answer.")

            except Exception as e:
                latency = time.perf_counter() - t0
                error = repr(e)

                if attempt < args.retries:
                    time.sleep(1.0)

        human_answers = get_human_answers(ex)
        canonical = str(ex["multiple_choice_answer"])

        score = (
            vqa_consensus_score(answer, human_answers)
            if not error
            else 0.0
        )

        canonical_exact = (
            normalize_answer(answer)
            == normalize_answer(canonical)
        ) if not error else False

        row = {
            "question_id": qid,
            "image_id": int(ex["image_id"]),
            "question_type": str(ex["question_type"]),
            "answer_type": str(ex["answer_type"]),
            "question": question,
            "canonical_answer": canonical,
            "human_answers": human_answers,
            "e2b_answer": answer,
            "vqa_score": float(score),
            "canonical_exact": bool(canonical_exact),
            "latency_s": float(latency) if latency is not None else None,
            "error": error,
        }

        # Write after EVERY case so Ctrl-C / crash is safe.
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

        completed[qid] = row

    # --------------------------------------------------------
    # Final summary using exactly this selected subset
    # --------------------------------------------------------

    completed = load_completed(output_path)

    final_rows = [
        completed[qid]
        for qid in selected_ids
        if qid in completed
    ]

    create_summary(final_rows, output_path)


if __name__ == "__main__":
    main()
