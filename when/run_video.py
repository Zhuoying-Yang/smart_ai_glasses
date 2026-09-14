#!/usr/bin/env python3
"""阶段 1：播放视频文件，同时在终端持续输出 WHEN 标签。

播放在主线程（macOS 要求 cv2 窗口在主线程），门在后台线程按固定采样率跑，
帧不够快就丢——这就是 Dispider 主张但它自己没实现的「反应不阻塞感知」。
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import os as _os

try:
    import cv2
except ImportError:
    raise SystemExit("missing opencv-python")

try:
    from .config import DEFAULT_CONFIG, load_config
    from .console import ConsoleSink
    from .encoder import SiglipEncoder
    from .gate import VisualGate
except ModuleNotFoundError as exc:
    _MISSING = exc.name
else:
    _MISSING = None




def _wrong_env(missing: str) -> "NoReturn":
    import sys as _s
    in_venv = _s.prefix != _s.base_prefix
    hint = ("  你在一个 venv 里：" + _s.prefix + "\n"
            "  先退出它：deactivate\n"
            if in_venv else
            "  先激活项目环境：conda activate eyewhen\n")
    _s.exit(
        f"\n✗ 缺少 {missing}，说明用错了 Python 解释器。\n"
        f"  当前：{_s.executable}\n"
        f"{hint}"
        f"  注意：拉起 Aria 桥接不需要激活 aria_env，本程序会用绝对路径调它。\n"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="播放视频并实时输出 WHEN 标签")
    p.add_argument("video", help="视频文件路径")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--model", help="覆盖配置里的 SigLIP 模型")
    p.add_argument("--negatives", choices=("off", "manual", "auto"),
                   help="覆盖负样本模式。新环境/摄像头对着自己时建议 auto")
    p.add_argument("--jsonl", help="把完整事件流写到这个文件")
    p.add_argument("--no-display", action="store_true", help="不开窗口，只出标签")
    p.add_argument("--speed", type=float, default=1.0, help="播放倍速（默认 1.0 实时）")
    p.add_argument("--fps", type=float, help="覆盖配置里的门采样率")
    p.add_argument("--min-raw", type=float, help="覆盖绝对下限（标定时设 0 可看全部原始分）")
    p.add_argument("--only", nargs="*", help="只启用这些 query id")
    p.add_argument("--deterministic", action="store_true",
                   help="按帧号采样、不丢帧、不按墙上时间播放。离线评测必须用这个")
    return p


class _Slot:
    """单槽缓冲：门来不及处理就直接丢旧帧，永不堆积。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.item = None
        self.closed = False

    def put(self, frame, t: float) -> None:
        with self.lock:
            self.item = (frame, t)

    def take(self):
        with self.lock:
            it, self.item = self.item, None
            return it


def _score_mode_label(cfg, gate) -> str:
    mode = cfg.negatives.mode
    if mode == "auto":
        return f"auto(词表自动挑，探测 {cfg.negatives.auto.probe_seconds:g}s)"
    if gate.cosine_mode:
        return "cosine(无负样本·零配置)"
    return f"manual({len(cfg.negatives.manual)} 条负样本)"


def _on_probe_done(gate) -> None:
    print(f"  ▣ 环境探测完成，自动选出 {len(gate.picked_negatives)} 条负样本：")
    for x in gate.picked_negatives:
        print(f"      - {x}")
    if gate._min_raw_override is not None:
        print(f"      标定 min_raw = {gate._min_raw_override:.3f}")
    print()


def main(argv=None) -> int:
    if _MISSING:
        _wrong_env(_MISSING)
    args = build_parser().parse_args(argv)
    video = Path(args.video)
    if not video.exists():
        print(f"找不到视频：{video}", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    if args.model:
        cfg.model.siglip = args.model
    if args.negatives:
        cfg.negatives.mode = args.negatives
    if args.fps:
        cfg.gate.fps = args.fps
    if args.min_raw is not None:
        cfg.gate.min_raw = args.min_raw
    if args.only:
        keep = set(args.only)
        cfg.queries = [q for q in cfg.queries if q.id in keep]
        if not cfg.queries:
            print(f"--only {args.only} 没匹配到任何 query", file=sys.stderr)
            return 2

    print(f"加载 SigLIP：{cfg.model.siglip} ...", flush=True)
    enc = SiglipEncoder(cfg.model)
    print(f"  device={enc.device} dtype={enc.dtype} 加载耗时={enc.load_s:.1f}s", flush=True)

    gate = VisualGate(cfg, enc)
    gate.on_probe_done = _on_probe_done
    print(f"  启用 {len(gate.queries)} 个视觉 query")
    for q in gate.queries:
        print(f"    [{q.trigger_type.value:8s}] {q.id:12s} {q.text}")
    _on, _off, _mr = gate.thresholds()
    _sm = _score_mode_label(cfg, gate)
    print(f"  门采样率 {cfg.gate.fps} FPS，mode={cfg.gate.mode}  打分={_sm}")
    print(f"  阈值 on={_on} off={_off} min_raw={_mr}\n")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"打不开视频：{video}", file=sys.stderr)
        return 2
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"视频：{video.name}  {src_fps:.2f} FPS  {n_frames} 帧  {n_frames/src_fps:.1f}s\n")

    jsonl_fp = open(args.jsonl, "w", encoding="utf-8") if args.jsonl else None
    sink = ConsoleSink(jsonl=jsonl_fp)
    slot = _Slot()

    def worker() -> None:
        interval = 1.0 / cfg.gate.fps
        next_at = 0.0
        while not slot.closed:
            item = slot.take()
            if item is None:
                time.sleep(0.005)
                continue
            frame, t = item
            if t < next_at:
                continue
            next_at = t + interval
            sink.emit(gate.step(frame, t))

    th = threading.Thread(target=worker, daemon=True)
    th.start()

    if args.deterministic:
        step = max(1, int(round(src_fps / cfg.gate.fps)))
        i = 0
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                if i % step == 0:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    sink.emit(gate.step(rgb, i / src_fps))
                    if not args.no_display:
                        cv2.imshow("WHEN — 按 q 退出", bgr)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                i += 1
        except KeyboardInterrupt:
            print("\n中断")
        finally:
            slot.closed = True
            th.join(timeout=2.0)
            cap.release()
            if not args.no_display:
                cv2.destroyAllWindows()
            if jsonl_fp:
                jsonl_fp.close()
        print(f"\n{'='*60}")
        print(f"结束（确定性）  {sink.summary()}")
        print(f"SigLIP 编码耗时  {enc.timing_report()}")
        if args.jsonl:
            print(f"事件流已写入  {args.jsonl}")
        return 0

    t_start = time.perf_counter()
    i = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            t_video = i / src_fps
            i += 1

            slot.put(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t_video)

            if not args.no_display:
                cv2.imshow("WHEN — 按 q 退出", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # 按真实时间轴播放
            target = t_start + t_video / max(args.speed, 1e-6)
            lag = target - time.perf_counter()
            if lag > 0:
                time.sleep(lag)
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        slot.closed = True
        th.join(timeout=2.0)
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        if jsonl_fp:
            jsonl_fp.close()

    print(f"\n{'='*60}")
    print(f"结束  {sink.summary()}")
    print(f"SigLIP 编码耗时  {enc.timing_report()}")
    if args.jsonl:
        print(f"事件流已写入  {args.jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
