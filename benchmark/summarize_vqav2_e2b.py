import json
from pathlib import Path

import pandas as pd


INPUT = Path("benchmark/results/vqav2_e2b_30k.jsonl")
OUT_DIR = Path("benchmark/results/final_10k")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load results
# ============================================================

rows_by_id = {}

with INPUT.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        try:
            row = json.loads(line)
            # In case the same question somehow appears twice,
            # keep the latest result.
            rows_by_id[str(row["question_id"])] = row
        except Exception:
            pass

rows = list(rows_by_id.values())

if not rows:
    raise RuntimeError("No valid rows found.")

df = pd.DataFrame(rows)

# Successful inference only
valid = df[
    df["error"].isna() |
    (df["error"] == "")
].copy()

print("=" * 78)
print("E2B VQAv2 FINAL BENCHMARK")
print("=" * 78)

print(f"Total completed cases : {len(df):,}")
print(f"Successful cases      : {len(valid):,}")
print(f"Inference errors      : {len(df) - len(valid):,}")
print(f"Question types        : {valid['question_type'].nunique()}")

if len(valid) == 0:
    raise RuntimeError("No successful inference results found.")


# ============================================================
# Overall metrics
# ============================================================

overall_vqa = valid["vqa_score"].mean()
overall_exact = valid["canonical_exact"].astype(float).mean()

mean_latency = valid["latency_s"].mean()
median_latency = valid["latency_s"].median()
p95_latency = valid["latency_s"].quantile(0.95)

print()
print("OVERALL")
print("-" * 78)
print(f"VQA consensus accuracy : {overall_vqa * 100:.2f}%")
print(f"Canonical exact acc    : {overall_exact * 100:.2f}%")
print(f"Mean latency           : {mean_latency:.3f} s")
print(f"Median latency         : {median_latency:.3f} s")
print(f"P95 latency            : {p95_latency:.3f} s")


# ============================================================
# Question-type summary
# ============================================================

qtype = (
    valid.groupby("question_type")
    .agg(
        n=("question_id", "count"),
        vqa_accuracy=("vqa_score", "mean"),
        canonical_exact_accuracy=("canonical_exact", "mean"),
        mean_latency_s=("latency_s", "mean"),
        median_latency_s=("latency_s", "median"),
    )
    .reset_index()
)

qtype["failure_rate"] = 1.0 - qtype["vqa_accuracy"]

qtype["vqa_accuracy_pct"] = qtype["vqa_accuracy"] * 100
qtype["failure_rate_pct"] = qtype["failure_rate"] * 100
qtype["canonical_exact_accuracy_pct"] = (
    qtype["canonical_exact_accuracy"] * 100
)

qtype = qtype.sort_values(
    ["failure_rate", "n"],
    ascending=[False, False]
)

qtype_path = OUT_DIR / "e2b_by_question_type.csv"
qtype.to_csv(qtype_path, index=False)


# ============================================================
# Answer-type summary
# ============================================================

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

atype["failure_rate"] = 1.0 - atype["vqa_accuracy"]

atype["vqa_accuracy_pct"] = atype["vqa_accuracy"] * 100
atype["failure_rate_pct"] = atype["failure_rate"] * 100
atype["canonical_exact_accuracy_pct"] = (
    atype["canonical_exact_accuracy"] * 100
)

atype_path = OUT_DIR / "e2b_by_answer_type.csv"
atype.to_csv(atype_path, index=False)


# ============================================================
# Save all successful cases
# ============================================================

valid_path = OUT_DIR / "e2b_all_10015_cases.csv"
valid.to_csv(valid_path, index=False)


# ============================================================
# Save wrong / low-score cases
# ============================================================

wrong = valid[
    valid["vqa_score"] < 1.0
].copy()

wrong = wrong.sort_values(
    ["vqa_score", "question_type"],
    ascending=[True, True]
)

wrong_path = OUT_DIR / "e2b_low_score_cases.csv"
wrong.to_csv(wrong_path, index=False)


# ============================================================
# Binary-style failure summary
#
# Strict failure:
# VQA score < 0.5
#
# Useful as a simple failure-rate statistic.
# ============================================================

valid["strict_failure"] = valid["vqa_score"] < 0.5

binary = (
    valid.groupby("question_type")
    .agg(
        n=("question_id", "count"),
        failures=("strict_failure", "sum"),
        mean_vqa_score=("vqa_score", "mean"),
    )
    .reset_index()
)

binary["strict_failure_rate"] = (
    binary["failures"] / binary["n"]
)

binary["strict_failure_rate_pct"] = (
    binary["strict_failure_rate"] * 100
)

binary = binary.sort_values(
    ["strict_failure_rate", "n"],
    ascending=[False, False]
)

binary_path = OUT_DIR / "e2b_strict_failure_by_question_type.csv"
binary.to_csv(binary_path, index=False)


# ============================================================
# Human-readable summary
# ============================================================

summary_path = OUT_DIR / "e2b_summary.txt"

with summary_path.open("w", encoding="utf-8") as f:
    f.write("E2B VQAv2 FINAL BENCHMARK\n")
    f.write("=" * 78 + "\n\n")

    f.write(f"Total completed cases : {len(df):,}\n")
    f.write(f"Successful cases      : {len(valid):,}\n")
    f.write(f"Inference errors      : {len(df) - len(valid):,}\n")
    f.write(
        f"Question types        : "
        f"{valid['question_type'].nunique()}\n\n"
    )

    f.write(
        f"VQA consensus accuracy : "
        f"{overall_vqa * 100:.2f}%\n"
    )
    f.write(
        f"Canonical exact acc    : "
        f"{overall_exact * 100:.2f}%\n"
    )
    f.write(
        f"Mean latency           : "
        f"{mean_latency:.3f} s\n"
    )
    f.write(
        f"Median latency         : "
        f"{median_latency:.3f} s\n"
    )
    f.write(
        f"P95 latency            : "
        f"{p95_latency:.3f} s\n\n"
    )

    f.write(
        "QUESTION TYPES SORTED BY VQA FAILURE RATE\n"
    )
    f.write("-" * 78 + "\n")

    for _, r in qtype.iterrows():
        f.write(
            f"{r['question_type']:<35} "
            f"n={int(r['n']):>4}  "
            f"acc={r['vqa_accuracy_pct']:6.2f}%  "
            f"failure={r['failure_rate_pct']:6.2f}%  "
            f"lat={r['mean_latency_s']:.2f}s\n"
        )


# ============================================================
# Print most difficult / easiest question types
# Only show types with >= 20 examples
# ============================================================

stable = qtype[qtype["n"] >= 20].copy()

print()
print("HARDEST QUESTION TYPES (n >= 20)")
print("-" * 78)

for _, r in stable.head(15).iterrows():
    print(
        f"{r['question_type']:<35} "
        f"n={int(r['n']):>4}  "
        f"acc={r['vqa_accuracy_pct']:6.2f}%  "
        f"failure={r['failure_rate_pct']:6.2f}%"
    )

print()
print("EASIEST QUESTION TYPES (n >= 20)")
print("-" * 78)

for _, r in stable.tail(15).sort_values(
    "vqa_accuracy",
    ascending=False
).iterrows():
    print(
        f"{r['question_type']:<35} "
        f"n={int(r['n']):>4}  "
        f"acc={r['vqa_accuracy_pct']:6.2f}%  "
        f"failure={r['failure_rate_pct']:6.2f}%"
    )


print()
print("=" * 78)
print("FILES SAVED")
print("=" * 78)
print(summary_path)
print(qtype_path)
print(atype_path)
print(binary_path)
print(valid_path)
print(wrong_path)
