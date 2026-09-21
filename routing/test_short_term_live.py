from __future__ import annotations

import argparse

from routing.short_term_memory import (
    LiveImageRecorder,
    RollingVisualBuffer,
)

from routing.temporal_memory import (
    execute_memory,
    execute_temporal,
)


def print_status(buffer):
    frames = buffer.all_frames()

    print()
    print("==============================")
    print("SHORT-TERM MEMORY STATUS")
    print("==============================")
    print(
        f"Frames stored: {len(frames)}"
    )

    if frames:
        print(
            f"Oldest frame age: "
            f"{frames[0].age():.1f}s"
        )
        print(
            f"Newest frame age: "
            f"{frames[-1].age():.1f}s"
        )

    print("==============================")
    print()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        default="./aria_live_frame.jpg",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
    )

    args = parser.parse_args()

    visual_buffer = RollingVisualBuffer(
        window_seconds=15.0,
    )

    recorder = LiveImageRecorder(
        buffer=visual_buffer,
        source_image_path=args.image,
        sample_fps=args.fps,
    )

    recorder.start()

    print()
    print("======================================")
    print("AI GLASSES SHORT-TERM ROUTING TEST")
    print("======================================")
    print(f"Watching: {args.image}")
    print(f"Sampling: {args.fps} FPS")
    print("Buffer window: 15 seconds")
    print()
    print("Commands:")
    print()
    print(
        "  status"
    )
    print(
        "  t: What did I just do with the cup?"
    )
    print(
        "  m: Where did I leave my keys?"
    )
    print(
        "  quit"
    )
    print()
    print(
        "Let the camera run for a few seconds "
        "before asking a question."
    )
    print()

    try:
        while True:
            text = input("> ").strip()

            if not text:
                continue

            if text.lower() in {
                "q",
                "quit",
                "exit",
            }:
                break

            if text.lower() == "status":
                print_status(
                    visual_buffer
                )
                continue

            if text.lower().startswith("t:"):
                question = text[2:].strip()

                if not question:
                    print(
                        "Please enter a temporal question."
                    )
                    continue

                result = execute_temporal(
                    question=question,
                    visual_buffer=visual_buffer,
                )

                print()
                print("ANSWER:")
                print(result["answer"])

                if result.get("success"):
                    print()
                    print(
                        f"Frames: "
                        f"{result['num_frames']}"
                    )
                    print(
                        f"Cloud latency: "
                        f"{result['cloud_latency']:.3f}s"
                    )
                    print(
                        f"End-to-end latency: "
                        f"{result['e2e_latency']:.3f}s"
                    )

                print()
                continue

            if text.lower().startswith("m:"):
                question = text[2:].strip()

                if not question:
                    print(
                        "Please enter a memory question."
                    )
                    continue

                result = execute_memory(
                    question=question,
                    visual_buffer=visual_buffer,
                )

                print()
                print("ANSWER:")
                print(result["answer"])

                if result.get("success"):
                    print()
                    print(
                        f"Frames: "
                        f"{result['num_frames']}"
                    )
                    print(
                        f"Cloud latency: "
                        f"{result['cloud_latency']:.3f}s"
                    )
                    print(
                        f"End-to-end latency: "
                        f"{result['e2e_latency']:.3f}s"
                    )

                print()
                continue

            print(
                "Use: status, t: QUESTION, "
                "m: QUESTION, or quit"
            )

    except KeyboardInterrupt:
        print()

    finally:
        recorder.stop()
        print(
            "Short-term recorder stopped."
        )


if __name__ == "__main__":
    main()
