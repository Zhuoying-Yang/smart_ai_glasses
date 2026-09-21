from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


@dataclass(frozen=True)
class FrameRecord:
    frame_id: int
    timestamp: float
    path: str

    def age(self, now=None):
        if now is None:
            now = time.time()
        return max(0.0, now - self.timestamp)


class RollingVisualBuffer:
    def __init__(
        self,
        window_seconds=15.0,
        storage_dir="routing/runtime/short_memory",
        disk_retention_seconds=90.0,
    ):
        self.window_seconds = float(window_seconds)
        self.disk_retention_seconds = float(disk_retention_seconds)

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._frames = deque()
        self._lock = threading.Lock()
        self._next_frame_id = 0

    def add_snapshot(self, source_path, timestamp=None):
        source = Path(source_path)

        if not source.exists():
            raise FileNotFoundError(source)

        if timestamp is None:
            timestamp = time.time()

        with self._lock:
            frame_id = self._next_frame_id
            self._next_frame_id += 1

        destination = self.storage_dir / (
            f"frame_{frame_id:08d}_{int(timestamp * 1000)}.jpg"
        )

        temp = self.storage_dir / (
            f".tmp_{frame_id:08d}.jpg"
        )

        # Decode then re-save so we never keep a partially written JPEG.
        with Image.open(source) as img:
            img.load()
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            img.save(temp, format="JPEG", quality=90)

        temp.replace(destination)

        record = FrameRecord(
            frame_id=frame_id,
            timestamp=timestamp,
            path=str(destination),
        )

        with self._lock:
            self._frames.append(record)
            self._prune_locked(timestamp)

        self._cleanup_disk(timestamp)

        return record

    def _prune_locked(self, now):
        cutoff = now - self.window_seconds

        while self._frames and self._frames[0].timestamp < cutoff:
            self._frames.popleft()

    def _cleanup_disk(self, now):
        cutoff = now - self.disk_retention_seconds

        for path in self.storage_dir.glob("frame_*.jpg"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except FileNotFoundError:
                pass

    def get_last(self, seconds):
        seconds = min(float(seconds), self.window_seconds)

        now = time.time()
        cutoff = now - seconds

        with self._lock:
            self._prune_locked(now)

            return [
                frame
                for frame in self._frames
                if frame.timestamp >= cutoff
            ]

    def uniform_sample(self, seconds, k):
        frames = self.get_last(seconds)

        if not frames or k <= 0:
            return []

        if len(frames) <= k:
            return frames

        if k == 1:
            return [frames[-1]]

        indices = [
            round(i * (len(frames) - 1) / (k - 1))
            for i in range(k)
        ]

        return [frames[i] for i in indices]

    def all_frames(self):
        return self.get_last(self.window_seconds)

    def __len__(self):
        with self._lock:
            return len(self._frames)


class LiveImageRecorder:
    def __init__(
        self,
        buffer,
        source_image_path,
        sample_fps=2.0,
    ):
        self.buffer = buffer
        self.source_image_path = Path(source_image_path)

        self.sample_fps = float(sample_fps)
        self.sample_period = 1.0 / self.sample_fps

        self._thread = None
        self._stop_event = threading.Event()
        self._last_signature = None

    def start(self):
        if self._thread is not None:
            return

        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _signature(self):
        stat = self.source_image_path.stat()
        return (stat.st_mtime_ns, stat.st_size)

    def _run(self):
        while not self._stop_event.is_set():

            if self.source_image_path.exists():
                try:
                    signature = self._signature()

                    # Save only when Aria produced a new frame.
                    if signature != self._last_signature:
                        self.buffer.add_snapshot(
                            str(self.source_image_path)
                        )
                        self._last_signature = signature

                except Exception as exc:
                    print(
                        f"[SHORT_MEMORY] skipped frame: {exc}"
                    )

            self._stop_event.wait(self.sample_period)
