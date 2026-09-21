import argparse
import time
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default="aria_live_frame.jpg",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
    )

    args = parser.parse_args()

    output = Path(args.output)

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {args.camera}"
        )

    period = 1.0 / args.fps

    print()
    print("==============================")
    print("LIVE CAMERA SOURCE")
    print("==============================")
    print(f"Camera index: {args.camera}")
    print(f"Output: {output}")
    print(f"FPS: {args.fps}")
    print()
    print("Press Ctrl+C to stop.")
    print()

    try:
        while True:
            start = time.perf_counter()

            ok, frame = cap.read()

            if not ok:
                print("Failed to read camera frame.")
                time.sleep(0.1)
                continue

            temp = output.with_name(
                "." + output.name + ".tmp.jpg"
            )

            ok = cv2.imwrite(
                str(temp),
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    90,
                ],
            )

            if ok:
                temp.replace(output)

            elapsed = (
                time.perf_counter() - start
            )

            time.sleep(
                max(0.0, period - elapsed)
            )

    except KeyboardInterrupt:
        print("\nStopping camera.")

    finally:
        cap.release()


if __name__ == "__main__":
    main()
