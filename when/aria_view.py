#!/usr/bin/env python3
"""Live preview of what the Aria glasses see. Nothing else.

    ~/aria_env/bin/python -m when.aria_view

Press q or ESC to quit.

Must be started with -m from the repo root: when/types.py shadows the stdlib
`types` module for any script launched by path from inside when/.
"""

from __future__ import annotations

import os
import sys as _sys

if os.path.dirname(os.path.abspath(__file__)) == os.path.abspath(_sys.path[0]):
    raise SystemExit(
        "Start this with -m from the repo root, not by path:\n"
        "  ~/aria_env/bin/python -m when.aria_view"
    )

import argparse
import contextlib
import select
import threading
import time

import cv2
import numpy as np


@contextlib.contextmanager
def terminal_keys():
    """让终端逐字符地交出按键，不用等回车。

    从终端启动的 Python 不是 macOS app bundle，它的 OpenCV 窗口成不了 key window
    —— 焦点拿不到，窗口的关闭按钮是灰的，按 q 也只会打进终端。
    所以干脆在终端这边收键。
    """
    if not _sys.stdin.isatty():
        yield lambda: None
        return
    import termios
    import tty

    fd = _sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)

        def poll():
            if select.select([_sys.stdin], [], [], 0)[0]:
                return _sys.stdin.read(1)
            return None

        yield poll
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _activate_app() -> None:
    """尽量把窗口顶到前台。装了 pyobjc 才有效，没有就算了。"""
    try:
        from AppKit import NSApp, NSApplication

        NSApplication.sharedApplication()
        NSApp.activateIgnoringOtherApps_(True)
    except Exception:
        pass

try:
    import aria.sdk as aria
except ImportError:
    raise SystemExit(
        "aria.sdk not found. Use the interpreter that has the SDK:\n"
        "  ~/aria_env/bin/python -m when.aria_view"
    )


class Viewer:
    """SDK pushes frames from its own thread; cv2 must draw on the main one."""

    def __init__(self, rotate: bool):
        self.rotate = rotate
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.count = 0
        self.t0 = time.perf_counter()
        self.fps = 0.0

    def on_image_received(self, image, record) -> None:  # noqa: ANN001
        frame = np.array(image, copy=True)
        if self.rotate:
            # Aria's RGB sensor is mounted rotated; raw frames come in sideways.
            frame = np.rot90(frame, -1).copy()
        with self.lock:
            self.frame = frame
            self.count += 1
            dt = time.perf_counter() - self.t0
            if dt >= 1.0:
                self.fps = self.count / dt
                self.count, self.t0 = 0, time.perf_counter()

    def take(self):
        with self.lock:
            return self.frame, self.fps


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Live preview of the Aria RGB camera")
    p.add_argument("--profile", default="profile12")
    p.add_argument("--interface", choices=("usb", "wifi"), default="usb")
    p.add_argument("--size", type=int, default=800, help="window size; 0 keeps 1408x1408")
    p.add_argument("--no-rotate", action="store_true")
    p.add_argument("--topmost", action="store_true",
                   help="keep the window above everything. Off by default: a topmost "
                        "window on macOS often refuses keyboard focus, so q stops working")
    p.add_argument("--serial")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    aria.set_log_level(aria.Level.Error)
    viewer = Viewer(not args.no_rotate)

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
    sub.message_queue_size[aria.StreamingDataType.Rgb] = 1
    sub.security_options.use_ephemeral_certs = True
    client.subscription_config = sub
    client.set_streaming_client_observer(viewer)
    client.subscribe()

    title = "Aria RGB"
    print(f"streaming {args.profile}, waiting for the first frame ...", flush=True)

    opened = False
    t_log = time.perf_counter()
    try:
      with terminal_keys() as poll_key:
        while True:
              frame, fps = viewer.take()
              if frame is None:
                  time.sleep(0.02)
                  continue

              if not opened:
                  # macOS: create the window explicitly and force it to the front,
                  # otherwise it can come up behind the terminal and look like nothing
                  # happened.
                  cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                  if args.size:
                      cv2.resizeWindow(title, args.size, args.size)
                  if args.topmost:
                      try:
                          cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
                      except Exception:
                          pass
                  opened = True
                  print(f"first frame {frame.shape[1]}x{frame.shape[0]} -- window opened.")
                  _activate_app()
                  print("Quit: press q HERE in this terminal (or Ctrl-C).", flush=True)
                  print("      The window itself may not take keyboard focus on macOS.\n",
                        flush=True)

              bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
              if args.size:
                  bgr = cv2.resize(bgr, (args.size, args.size), interpolation=cv2.INTER_AREA)
              cv2.putText(bgr, f"{frame.shape[1]}x{frame.shape[0]}  {fps:4.1f} fps",
                          (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
              cv2.imshow(title, bgr)

              # 终端也报一下，窗口万一藏起来了也能看出流是活的
              now = time.perf_counter()
              if now - t_log >= 3.0:
                  print(f"  {fps:4.1f} fps", flush=True)
                  t_log = now

              # 窗口里的按键（焦点正常时才收得到）
              if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                  break
              # 终端里的按键 —— 这条在 macOS 上才是真正管用的那个
              key = poll_key()
              if key in ("q", "Q", "\x1b"):
                  break
              # 点窗口的关闭按钮时 OpenCV 不会通知，得自己查它还在不在
              try:
                  if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                      print("window closed")
                      break
              except cv2.error:
                  break
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        cv2.destroyAllWindows()
        for _ in range(5):          # macOS 需要再泵几次事件窗口才真的关掉
            cv2.waitKey(1)
        for step in (client.unsubscribe, manager.stop_streaming,
                     lambda: device_client.disconnect(device)):
            try:
                step()
            except Exception as exc:
                print(f"  cleanup warning: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
