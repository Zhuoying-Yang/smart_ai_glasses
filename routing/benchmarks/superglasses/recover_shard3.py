from pathlib import Path
from io import BytesIO

import pandas as pd
from huggingface_hub import hf_hub_download
from datasets import load_dataset, Image as HFImage
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

MANIFEST = Path(
    "routing/benchmarks/superglasses/"
    "full_routing_manifest.csv"
)

OUT_DIR = Path(
    "routing/benchmarks/superglasses/images"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


df = pd.read_csv(MANIFEST)

needed = set(
    df["image_id"]
    .astype(int)
    .tolist()
)

saved = set()

for p in OUT_DIR.glob("*.jpg"):
    try:
        saved.add(int(p.stem))
    except ValueError:
        pass

missing = needed - saved

print("=" * 90)
print("RECOVER SUPERGLASSES SHARD 3")
print("=" * 90)
print("Required :", len(needed))
print("Saved    :", len(saved & needed))
print("Missing  :", len(missing))

if not missing:
    print("Nothing to recover.")
    raise SystemExit(0)


print()
print("Downloading only shard 3...")

parquet_path = hf_hub_download(
    repo_id="xandery/SuperGlasses",
    repo_type="dataset",
    filename=(
        "images/"
        "test-00003-of-00018.parquet"
    ),
)

print("Downloaded:")
print(parquet_path)


ds = load_dataset(
    "parquet",
    data_files={
        "train": parquet_path,
    },
    split="train",
)

ds = ds.cast_column(
    "image",
    HFImage(decode=False),
)


recovered = 0

for row in ds:

    image_id = int(
        row["image_id"]
    )

    if image_id not in missing:
        continue

    img = row["image"]

    try:
        if img.get("bytes") is not None:
            pil = Image.open(
                BytesIO(img["bytes"])
            )

        elif img.get("path"):
            pil = Image.open(
                img["path"]
            )

        else:
            print(
                "No bytes/path:",
                image_id,
            )
            continue

        pil.load()
        pil = pil.convert("RGB")

        out = (
            OUT_DIR
            / f"{image_id}.jpg"
        )

        pil.save(
            out,
            format="JPEG",
            quality=95,
        )

        recovered += 1

        print(
            f"SAVED {image_id} "
            f"({recovered}/{len(missing)})"
        )

    except Exception as e:
        print(
            f"WARNING {image_id}: "
            f"{type(e).__name__}: {e}"
        )


saved_after = set()

for p in OUT_DIR.glob("*.jpg"):
    try:
        saved_after.add(int(p.stem))
    except ValueError:
        pass

remaining = needed - saved_after


print()
print("=" * 90)
print("RECOVERY SUMMARY")
print("=" * 90)

print(
    "Recovered       :",
    recovered,
)

print(
    "Total available :",
    len(needed & saved_after),
)

print(
    "Still missing   :",
    len(remaining),
)

if remaining:
    print(
        "Missing IDs:",
        sorted(remaining),
    )
else:
    print(
        "All SuperGlasses images ready."
    )
