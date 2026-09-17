from pathlib import Path

import pandas as pd
from datasets import load_dataset


OUT_DIR = Path(
    "routing/benchmarks/wearvqa_full"
)

IMG_DIR = OUT_DIR / "images"
MANIFEST = OUT_DIR / "manifest.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)


print("Loading full WearVQA...")

ds = load_dataset(
    "tonyliao-meta/WearVQA",
    split="train",
)

print("Cases:", len(ds))
print("Columns:", ds.column_names)
print()


rows = []

for i, sample in enumerate(ds):

    # HF WebDataset structure:
    #   jpg  -> PIL image
    #   json -> metadata dict
    image = sample["jpg"]
    meta = sample["json"]

    sample_id = meta.get(
        "idx",
        i,
    )

    image_path = (
        IMG_DIR / f"{sample_id}.jpg"
    )

    if not image_path.exists():
        image.convert("RGB").save(
            image_path,
            quality=95,
        )

    # Be tolerant to naming differences.
    question = meta.get(
        "question",
        ""
    )

    ground_truth = (
        meta.get("response")
        or meta.get("ground_truth")
        or meta.get("answer")
        or meta.get("reference_answer")
        or ""
    )

    category = (
        meta.get("question_type")
        or meta.get("category")
        or meta.get("task_type")
        or ""
    )

    row = {
        "sample_id": sample_id,
        "image": str(image_path),
        "question": question,
        "ground_truth": ground_truth,
        "category": category,
        "domain": meta.get("domain"),

        "is_not_zoomed_in":
            meta.get("is_not_zoomed_in"),

        "is_leveling":
            meta.get("is_leveling"),

        "is_cut_off":
            meta.get("is_cut_off"),

        "is_blur":
            meta.get("is_blur"),

        "is_low_light":
            meta.get("is_low_light"),

        "is_occluded":
            meta.get("is_occluded"),

        "hand_finger_elements":
            meta.get("hand_finger_elements"),
    }

    rows.append(row)

    if (i + 1) % 250 == 0:
        print(
            f"Prepared {i+1}/{len(ds)}"
        )


df = pd.DataFrame(rows)

df.to_csv(
    MANIFEST,
    index=False,
)

print()
print("=" * 72)
print("WEARVQA FULL PREPARED")
print("=" * 72)

print("Rows:", len(df))
print("Categories:", df["category"].nunique())
print("Domains:", df["domain"].nunique())

print()
print(
    df[
        [
            "sample_id",
            "question",
            "ground_truth",
            "category",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print()
print("Saved:")
print(MANIFEST)
