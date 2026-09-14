#!/usr/bin/env python3
"""Project Aria glasses -> local MJPEG stream.

Aria does not enumerate as a UVC camera, so cv2.VideoCapture cannot see it.
Frames only come through the Aria Client SDK, whose dependency tree conflicts
with this project's environment. So this script runs in its own interpreter
and republishes the RGB stream as MJPEG over localhost, which OpenCV reads
natively:

    # terminal 1 (aria env)
    ~/aria_env/bin/python -m when.aria_bridge

    # terminal 2 (project env)
    python -m when.run_live --camera http://127.0.0.1:8080/ --no-preset --negatives auto

Run it with the interpreter that has projectaria_client_sdk installed. It
imports nothing from the `when` package, but must be started with `-m` from
the repo root: `when/types.py` shadows the stdlib `types` module for any
script launched by path from inside `when/`.
"""

from __future__ import annotations

import os
import sys as _sys

if os.path.dirname(os.path.abspath(__file__)) == os.path.abspath(_sys.path[0]):
    raise SystemExit(
        "Start this with -m from the repo root, not by path:\n"
        "  ~/aria_env/bin/python -m when.aria_bridge\n"
        "Reason: when/types.py shadows the stdlib `types` module when when/ "
        "is on sys.path[0], which breaks imports deep inside the stdlib."
    )

import argparse
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

try:
    import aria.sdk as aria
except ImportError:
    sys.exit(
        "aria.sdk not found. Run this with the interpreter that has the SDK:\n"
        "  ~/aria_env/bin/python -m when.aria_bridge"
    )

BOUNDARY = "ariaframe"


class LatestFrame:
    """Single-slot buffer: a slow consumer drops frames instead of lagging."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._seq = 0
        self._new = threading.Condition(self._lock)

    def put(self, jpeg: bytes) -> None:
        with self._new:
            self._jpeg = jpeg
            self._seq += 1
            self._new.notify_all()

    def wait_next(self, last_seq: int, timeout: float = 5.0):
        with self._new:
            if self._seq == last_seq:
                self._new.wait(timeout)
            return self._jpeg, self._seq


class RgbObserver:
    """Aria SDK calls this on every RGB frame."""

    def __init__(self, buffer: LatestFrame, size: int, quality: int, rotate: bool):
        self.buffer = buffer
        self.size = size
        self.quality = quality
        self.rotate = rotate
        self.count = 0
        self.t0 = time.perf_counter()

    def on_image_received(self, image, record) -> None:  # noqa: ANN001
        frame = np.array(image, copy=True)
        if self.rotate:
            # Aria's RGB sensor is mounted rotated: raw frames come in on their
            # side. Verified against a known-upright object.
            frame = np.rot90(frame, -1).copy()
        if self.size and frame.shape[0] != self.size:
            frame = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(
            ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), self.quality],
        )
        if ok:
            self.buffer.put(buf.tobytes())
            self.count += 1


def make_handler(buffer: LatestFrame):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *args):        # 静音，别刷屏
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header(
                "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}"
            )
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            seq = -1
            try:
                while True:
                    jpeg, seq = buffer.wait_next(seq)
                    if jpeg is None:
                        continue
                    self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass                          # 消费端断开，正常

    return Handler


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Republish the Aria RGB stream as local MJPEG")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="127.0.0.1", help="127.0.0.1 keeps it off the network")
    p.add_argument("--profile", default="profile12",
                   help="profile12 = RGB 10fps 2MP (no audio); profile18 adds spatial audio")
    p.add_argument("--interface", choices=("usb", "wifi"), default="usb")
    p.add_argument("--size", type=int, default=640,
                   help="downscale the 1408x1408 frame before encoding; 0 keeps full size")
    p.add_argument("--quality", type=int, default=80, help="JPEG quality")
    p.add_argument("--no-rotate", action="store_true",
                   help="skip the 90 deg correction (raw frames are on their side)")
    p.add_argument("--serial", help="device serial, needed only with several devices attached")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    aria.set_log_level(aria.Level.Error)

    buffer = LatestFrame()
    observer = RgbObserver(buffer, args.size, args.quality, not args.no_rotate)

    device_client = aria.DeviceClient()
    cfg = aria.DeviceClientConfig()
    if args.serial:
        cfg.device_serial = args.serial
    device_client.set_client_config(cfg)

    print("connecting over USB ...", flush=True)
    device = device_client.connect()

    manager = device.streaming_manager
    config = manager.streaming_config
    config.profile_name = args.profile
    config.streaming_interface = (
        aria.StreamingInterface.Usb if args.interface == "usb"
        else aria.StreamingInterface.WifiStation
    )
    config.security_options.use_ephemeral_certs = True
    manager.streaming_config = config
    manager.start_streaming()

    client = manager.streaming_client
    sub = client.subscription_config
    sub.subscriber_data_type = aria.StreamingDataType.Rgb
    sub.message_queue_size[aria.StreamingDataType.Rgb] = 1   # 只留最新一帧
    sub.security_options.use_ephemeral_certs = True
    client.subscription_config = sub
    client.set_streaming_client_observer(observer)
    client.subscribe()

    server = ThreadedHTTPServer((args.host, args.port), make_handler(buffer))
    threading.Thread(target=server.serve_forever, daemon=True).start()

    url = f"http://{args.host}:{args.port}/"
    out = f"{args.size}x{args.size}" if args.size else "1408x1408"
    print(f"streaming  {args.profile}  {out}  q={args.quality}  rotate={not args.no_rotate}")
    print(f"MJPEG ->  {url}")
    print(f"consume:  python -m when.run_live --camera {url} --no-preset --negatives auto")
    print("Ctrl-C to stop.\n", flush=True)

    try:
        last, t_last = 0, time.perf_counter()
        while True:
            time.sleep(5.0)
            now = time.perf_counter()
            fps = (observer.count - last) / (now - t_last)
            print(f"  {observer.count} frames  ({fps:.1f} fps)", flush=True)
            last, t_last = observer.count, now
    except KeyboardInterrupt:
        print("\nstopping ...")
    finally:
        for step in (client.unsubscribe, manager.stop_streaming,
                     lambda: device_client.disconnect(device), server.shutdown):
            try:
                step()
            except Exception as exc:
                print(f"  cleanup warning: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
