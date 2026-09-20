from pathlib import Path
from io import BytesIO

import pandas as pd
from datasets import (
    load_dataset,
    Image as HFImage,
)
from PIL import Image, ImageFile


# Allow slightly truncated JPEGs.
ImageFile.LOAD_TRUNCATED_IMAGES = True


MANIFEST = Path(
    "routing/benchmarks/superglasses/"
    "direct_visual_manifest.csv"
)

OUT_DIR = Path(
    "routing/benchmarks/superglasses/images"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# IMPORTANT:
# shard 3 is currently timing out.
# Start directly at shard 4.
# ============================================================

SHARDS = list(range(4, 18))


df = pd.read_csv(MANIFEST)

needed = set(
    df["image_id"]
    .astype(int)
    .tolist()
)


def get_saved():
    ids = set()

    for p in OUT_DIR.glob("*.jpg"):
        try:
            ids.add(int(p.stem))
        except ValueError:
            pass

    return ids


def save_image(image_id, img):
    if isinstance(img, dict):

        if img.get("bytes") is not None:
            pil = Image.open(
                BytesIO(img["bytes"])
            )

        elif img.get("path"):
            pil = Image.open(
                img["path"]
            )

        else:
            raise RuntimeError(
                "Image has no bytes or path"
            )

    elif isinstance(img, Image.Image):
        pil = img

    else:
        raise RuntimeError(
            f"Unknown image type: {type(img)}"
        )

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


saved = get_saved()

print("=" * 100)
print("SUPERGLASSES SHARD-BY-SHARD DOWNLOAD")
print("=" * 100)

print("Required       :", len(needed))
print("Already saved :", len(saved & needed))
print("Still missing :", len(needed - saved))
print()
print("Skipping shard 3 for now.")
print("Processing shards 4 through 17.")
print()


failed_shards = []


for shard_idx in SHARDS:

    saved = get_saved()
    remaining = needed - saved

    if not remaining:
        break

    shard_name = (
        f"test-{shard_idx:05d}-of-00018.parquet"
    )

    uri = (
        "hf://datasets/"
        "xandery/SuperGlasses/"
        "images/"
        + shard_name
    )

    print()
    print("=" * 100)
    print(
        f"SHARD {shard_idx}/17"
    )
    print("=" * 100)

    print("File      :", shard_name)
    print("Remaining :", len(remaining))

    try:
        ds = load_dataset(
            "parquet",
            data_files={
                "test": uri
            },
            split="test",
            streaming=True,
        )

        # Don't let HF/PIL decode before we can handle errors.
        ds = ds.cast_column(
            "image",
            HFImage(decode=False),
        )

        scanned = 0
        new_saved = 0

        for row in ds:

            scanned += 1

            if scanned % 100 == 0:
                print(
                    f"  scanned={scanned:<5} "
                    f"new_saved={new_saved:<4} "
                    f"remaining={len(needed - get_saved())}"
                )

            image_id = int(
                row["image_id"]
            )

            # Not one of our 969 target images.
            if image_id not in needed:
                continue

            # Already downloaded on a previous run.
            if (
                OUT_DIR
                / f"{image_id}.jpg"
            ).exists():
                continue

            try:
                save_image(
                    image_id,
                    row["image"],
                )

                new_saved += 1

                print(
                    f"  SAVED {image_id} "
                    f"(new in shard={new_saved})"
                )

            except Exception as e:

                print(
                    f"  WARNING image {image_id}: "
                    f"{type(e).__name__}: {e}"
                )

                continue

        print()
        print(
            f"Finished shard {shard_idx}: "
            f"scanned={scanned}, "
            f"new_saved={new_saved}"
        )

    except Exception as e:

        failed_shards.append(
            shard_idx
        )

        print()
        print(
            f"WARNING: skipping shard "
            f"{shard_idx}"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        continue


# ============================================================
# Final summary
# ============================================================

saved = get_saved()

missing = (
    needed - saved
)


print()
print("=" * 100)
print("DOWNLOAD SUMMARY")
print("=" * 100)

print(
    "Required images :",
    len(needed),
)

print(
    "Saved images    :",
    len(needed & saved),
)

print(
    "Missing images  :",
    len(missing),
)

print(
    "Failed shards   :",
    failed_shards,
)

if missing:

    print()
    print("First 50 missing image IDs:")

    print(
        sorted(missing)[:50]
    )

else:

    print()
    print(
        "All direct-visual images are ready."
    )
