from pathlib import Path
from io import BytesIO

import pandas as pd
from datasets import load_dataset, Image as HFImage
from PIL import Image, ImageFile
from tqdm import tqdm


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

# Some SuperGlasses JPEGs are slightly truncated.
# PIL can usually still decode them safely.
ImageFile.LOAD_TRUNCATED_IMAGES = True


df = pd.read_csv(MANIFEST)

needed = set(
    df["image_id"]
    .astype(int)
    .tolist()
)

print("=" * 100)
print("SUPERGLASSES SELECTIVE IMAGE DOWNLOAD")
print("=" * 100)

print("Questions      :", len(df))
print("Unique images  :", len(needed))
print("Already saved  :", end=" ")


# ------------------------------------------------------------
# Check already-downloaded images
# ------------------------------------------------------------

saved_ids = set()

for p in OUT_DIR.glob("*"):
    try:
        saved_ids.add(
            int(p.stem)
        )
    except ValueError:
        pass

print(len(saved_ids & needed))

remaining = (
    needed - saved_ids
)

print("Remaining      :", len(remaining))

if not remaining:
    print("\nAll required images already exist.")
    raise SystemExit(0)


# ------------------------------------------------------------
# Stream the image config rather than materializing the
# complete ~8.8 GB dataset locally.
# ------------------------------------------------------------

print()
print("Opening SuperGlasses image stream...")

ds = load_dataset(
    "xandery/SuperGlasses",
    name="images",
    split="test",
    streaming=True,
)

# IMPORTANT:
# Prevent Hugging Face from automatically decoding images.
# Otherwise one truncated JPEG crashes the whole iterator
# before we have a chance to handle it.
ds = ds.cast_column(
    "image",
    HFImage(decode=False),
)


found = set()

progress = tqdm(
    total=len(remaining),
    desc="Saving selected images",
)


for row in ds:

    image_id = int(
        row["image_id"]
    )

    if image_id not in remaining:
        continue

    img = row["image"]

    try:
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
                    f"No bytes/path for image {image_id}"
                )

        elif isinstance(img, Image.Image):
            pil = img

        else:
            raise RuntimeError(
                f"Unknown image type for "
                f"{image_id}: {type(img)}"
            )

        # Force actual decoding here, under our control.
        pil.load()
        pil = pil.convert("RGB")

    except Exception as e:
        print()
        print(
            f"WARNING: skipping image {image_id}: "
            f"{type(e).__name__}: {e}"
        )
        continue

    out = (
        OUT_DIR
        / f"{image_id}.jpg"
    )

    pil.save(
        out,
        format="JPEG",
        quality=95,
    )

    found.add(image_id)
    progress.update(1)

    if len(found) == len(remaining):
        break


progress.close()


# ------------------------------------------------------------
# Verify
# ------------------------------------------------------------

all_saved = set()

for p in OUT_DIR.glob("*.jpg"):
    try:
        all_saved.add(
            int(p.stem)
        )
    except ValueError:
        pass


missing = (
    needed - all_saved
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
    len(needed - missing),
)

print(
    "Missing images  :",
    len(missing),
)

if missing:
    print(
        "First missing IDs:",
        sorted(missing)[:20],
    )
else:
    print(
        "All required SuperGlasses "
        "images are ready."
    )
