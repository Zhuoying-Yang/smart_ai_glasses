from pathlib import Path
import pandas as pd


ROOT = Path("routing/benchmarks")

OUT = (
    ROOT
    / "unified_question_corpus.csv"
)

rows = []


# ============================================================
# 1. WearVQA
# All questions are direct visual wearable VQA.
# ============================================================

wear_path = (
    ROOT
    / "wearvqa_full"
    / "manifest.csv"
)

wear = pd.read_csv(wear_path)

for _, r in wear.iterrows():

    rows.append({
        "source": "WearVQA",
        "sample_id": r["sample_id"],
        "question": r["question"],
        "ground_truth": r.get(
            "ground_truth"
        ),

        "routing_type":
            "DIRECT_VISUAL",

        "subtype":
            r.get("category"),

        "domain":
            r.get("domain"),
    })


# ============================================================
# 2. SuperGlasses
#
# Already classified as:
# DIRECT_VISUAL / KNOWLEDGE / TEMPORAL
# ============================================================

sg_path = (
    ROOT
    / "superglasses"
    / "full_routing_manifest.csv"
)

sg = pd.read_csv(sg_path)

for _, r in sg.iterrows():

    rows.append({
        "source": "SuperGlasses",
        "sample_id": r["sample_id"],
        "question": r["question"],
        "ground_truth": r.get(
            "ground_truth"
        ),

        "routing_type":
            r["routing_type"],

        "subtype":
            r.get("categories"),

        "domain":
            r.get("domain"),
    })


# ============================================================
# 3. SuperMemory-VQA
# ============================================================

sm_path = (
    ROOT
    / "supermemory"
    / "question_manifest.csv"
)

sm = pd.read_csv(sm_path)

for _, r in sm.iterrows():

    rows.append({
        "source":
            "SuperMemory-VQA",

        "sample_id":
            r["sample_id"],

        "question":
            r["question"],

        "ground_truth":
            None,

        "routing_type":
            "MEMORY",

        "subtype":
            r["subtype"],

        "domain":
            None,
    })


# ============================================================
# 4. EgoWearBench
#
# TEMPORAL + PROACTIVE
# ============================================================

ego_path = (
    ROOT
    / "egowearbench"
    / "question_manifest.csv"
)

ego = pd.read_csv(ego_path)

for _, r in ego.iterrows():

    rows.append({
        "source":
            "EgoWearBench",

        "sample_id":
            r["sample_id"],

        "question":
            r["question"],

        "ground_truth":
            r.get("ground_truth"),

        "routing_type":
            r["routing_type"],

        "subtype":
            r["subtype"],

        "domain":
            r.get("category"),
    })


# ============================================================
# Build dataframe
# ============================================================

df = pd.DataFrame(rows)

df["question"] = (
    df["question"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[
    df["question"] != ""
].copy()


# ============================================================
# Detect duplicates
# ============================================================

df["question_norm"] = (
    df["question"]
    .str.lower()
    .str.replace(
        r"\s+",
        " ",
        regex=True,
    )
    .str.strip()
)

duplicate_count = (
    df["question_norm"]
    .duplicated()
    .sum()
)


# ============================================================
# Save
# ============================================================

df.to_csv(
    OUT,
    index=False,
)


# ============================================================
# Summary
# ============================================================

print("=" * 100)
print("UNIFIED AI-GLASSES QUESTION CORPUS")
print("=" * 100)

print(
    "Total question rows :",
    len(df),
)

print(
    "Unique questions    :",
    df["question_norm"].nunique(),
)

print(
    "Duplicate rows      :",
    duplicate_count,
)


print()
print("=" * 100)
print("HIGH-LEVEL ROUTING TYPE")
print("=" * 100)

print(
    df["routing_type"]
    .value_counts()
    .to_string()
)


print()
print("=" * 100)
print("BY SOURCE")
print("=" * 100)

print(
    df["source"]
    .value_counts()
    .to_string()
)


print()
print("=" * 100)
print("SOURCE × ROUTING TYPE")
print("=" * 100)

table = pd.crosstab(
    df["source"],
    df["routing_type"],
)

print(
    table.to_string()
)


print()
print("=" * 100)
print("SUBTYPE DISTRIBUTION")
print("=" * 100)

print(
    df.groupby(
        ["routing_type", "subtype"]
    )
    .size()
    .sort_values(
        ascending=False
    )
    .head(50)
    .to_string()
)


print()
print("=" * 100)
print("SAVED")
print("=" * 100)

print(OUT)
