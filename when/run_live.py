#!/usr/bin/env python3
"""阶段 2：摄像头持续监测 + 麦克风提问，终端持续输出 WHEN 标签。

三条通路同时跑：
  摄像头 -> 视觉门 -> STANDING / ALERT 标签
  麦克风 -> VAD -> ASR -> INSTANT 标签，或当场注册一条新的 STANDING
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import threading
import time

import os as _os

try:
    import cv2
except ImportError:
    raise SystemExit("missing opencv-python")

from .asr import (
    MicStream,
    is_hallucination,
    RmsVad,
    Transcriber,
    classify_intent,
    to_visual_prompt,
    voice_command,
)
try:
    from .config import DEFAULT_CONFIG, QuerySpec, load_config
    from .console import ConsoleSink
    from .encoder import SiglipEncoder
    from .gate import VisualGate, emit_instant
    from .run_video import _Slot
    from .types import TriggerType, Urgency
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
    p = argparse.ArgumentParser(description="摄像头 + 语音，实时输出 WHEN 标签")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--model", help="覆盖配置里的 SigLIP 模型")
    p.add_argument("--negatives", choices=("off", "manual", "auto"),
                   help="覆盖负样本模式。新环境/摄像头对着自己时建议 auto")
    p.add_argument("--source", choices=("mac", "aria"),
                   help="画面来源：mac=电脑摄像头，aria=眼镜。都不给且在终端里跑时会让你选")
    p.add_argument("--camera", default=None,
                   help="摄像头编号，或 MJPEG/RTSP 地址。给了它就不再询问来源")
    p.add_argument("--aria-python", default="~/aria_env/bin/python",
                   help="装了 projectaria_client_sdk 的解释器")
    p.add_argument("--aria-port", type=int, default=8080, help="Aria 桥接的本地端口")
    p.add_argument("--aria-size", type=int, default=640, help="Aria 桥接下采样到的边长")
    p.add_argument("--jsonl", help="把完整事件流写到这个文件")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--no-audio", action="store_true", help="只跑视觉，不开麦克风")
    p.add_argument("--fps", type=float, help="覆盖配置里的门采样率")
    p.add_argument("--asr-model", default="mlx-community/whisper-small-mlx",
                   help="默认 small：又快又支持中译英。turbo 不支持 translate")
    p.add_argument("--language", default=None, help="ASR 语言，如 zh / en；不给则自动判别")
    p.add_argument("--vad-threshold", type=float, default=0.006,
                   help="VAD 能量下限。启动时会按环境底噪自动抬高，这只是地板值")
    p.add_argument("--mic", type=int, help="麦克风设备号；不给用系统默认")
    p.add_argument("--list-devices", action="store_true", help="列出音视频设备后退出")
    p.add_argument("--mic-test", action="store_true",
                   help="只开麦克风，实时显示音量和 VAD 判定。说话听不到时先跑这个")
    p.add_argument("--no-preset", action="store_true",
                   help="不加载 yaml 里的 standing/alerts，只监测你口头注册的（笔记本摄像头建议开）")
    p.add_argument("--only", nargs="*", help="只启用这些 query id")
    p.add_argument("--no-translate", action="store_true",
                   help="不把语音翻成英文。默认会翻——SigLIP 文本塔是英文训练的")
    return p


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


def _list_devices() -> int:
    import sounddevice as sd

    print("=== 麦克风 ===")
    default_in = sd.default.device[0]
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            mark = "  <<< 默认" if i == default_in else ""
            print(f"  [{i}] {d['name']}  sr={int(d['default_samplerate'])}{mark}")
    print("\n=== 摄像头 ===")
    for i in range(4):
        cap = cv2.VideoCapture(i)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if ok:
            print(f"  [{i}] 可用")
    return 0


def _mic_test(device=None, threshold: float = 0.012) -> int:
    """实时音量表。说话时看条形有没有越过 │ 那道阈值线。"""
    vad = RmsVad(threshold=threshold)
    mic = MicStream(vad, device=device)
    mic.start()
    print(f"麦克风：{mic.device_name}")
    print(f"先静音 {vad.calibrate_s:g} 秒量底噪，然后正常说话。Ctrl-C 退出。\n")
    n_ok = 0
    seen_rej = 0
    try:
        while True:
            time.sleep(0.1)
            r = vad.last_rms
            if not vad.calibrated:
                print(f"\r  标定中... rms={r:.4f}", end="", flush=True)
                continue
            thr = vad.threshold
            # 0 ~ 4×阈值 映射到 40 格，阈值落在第 10 格
            n = min(40, int(r / max(thr, 1e-6) * 10))
            bar = "█" * n + "·" * (40 - n)
            bar = bar[:10] + "│" + bar[10:]
            state = "说话中" if vad._speaking else "      "
            while not mic.out.empty():
                u = mic.out.get()
                n_ok += 1
                print(f"\r\033[K  \033[32m✓ 第 {n_ok} 句\033[0m  "
                      f"有效 {u.t_end - u.t_start:.2f}s  均值rms={u.rms:.4f}")
            if vad.rejected > seen_rej:
                seen_rej = vad.rejected
                print(f"\r\033[K  \033[33m✗ 拒收 #{seen_rej}\033[0m  {vad.last_reject}")
            print(f"\r\033[K  {bar}  rms={r:.4f}  阈值={thr:.4f}  峰值={vad.peak_rms:.4f}  {state}",
                  end="", flush=True)
    except KeyboardInterrupt:
        print("\n")
    finally:
        mic.stop()
    print(f"结果：收到 {n_ok} 句，拒收 {vad.rejected} 次，音量峰值 {vad.peak_rms:.4f}，阈值 {vad.threshold:.4f}")
    if n_ok:
        print("→ 麦克风正常，直接跑 run_live 就行")
    elif vad.rejected:
        print(f"→ 声音进来了但被判掉：{vad.last_reject}")
        print("   说长一点、大声一点，或者用 --vad-threshold 调低阈值")
    elif vad.peak_rms < vad.threshold:
        print(f"→ 音量峰值 {vad.peak_rms:.4f} 从没越过阈值 {vad.threshold:.4f}。")
        print(f"   麦克风离得太远，或者系统输入音量太低（系统设置 → 声音 → 输入）。")
        print(f"   临时绕过：--vad-threshold {max(vad.peak_rms * 0.5, 0.002):.4f}")
    else:
        print(f"→ 有瞬间越过阈值（峰值 {vad.peak_rms:.4f}），但没连成一句。")
        print(f"   需要连续 {vad.onset_blocks} 块(={vad.onset_blocks * 0.05:.2f}s)够响才算起音，"
              f"且整句不短于 {vad.min_speech_s:g}s。")
    return 0


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


def _choose_source() -> str:
    print("\n画面来源：")
    print("  1) 电脑摄像头")
    print("  2) 眼镜摄像头（Project Aria）")
    while True:
        try:
            ans = input("选择 [1/2]: ").strip()
        except EOFError:
            return "mac"
        if ans in ("1", "mac", ""):
            return "mac"
        if ans in ("2", "aria"):
            return "aria"
        print("  输入 1 或 2")


def _start_aria_bridge(args):
    """把桥接拉起来并等它就绪。返回 (子进程, URL)。

    Aria SDK 的依赖和本项目冲突，只能活在自己的解释器里，所以这里起一个子进程，
    由它把 RGB 流转成本地 MJPEG。已经有人在跑桥接就直接复用，不再起第二个
    —— 眼镜同一时间只允许一个流式会话。
    """
    url = f"http://127.0.0.1:{args.aria_port}/"
    if _port_open(args.aria_port):
        print(f"  ✓ 复用已在运行的 Aria 桥接 {url}")
        return None, url

    exe = os.path.expanduser(args.aria_python)
    if not os.path.exists(exe):
        print(f"\n✗ 找不到 Aria 解释器：{exe}\n"
              f"  先按 when/README.md 第 4.5 节建好 ~/aria_env，或用 --aria-python 指定\n",
              file=sys.stderr)
        return None, None

    print(f"  启动 Aria 桥接（{exe}）...")
    print("  眼镜端起流约需 15 秒，请稍候", flush=True)
    proc = subprocess.Popen(
        [exe, "-m", "when.aria_bridge", "--port", str(args.aria_port),
         "--size", str(args.aria_size)],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    for _ in range(600):                     # 最多等 60 秒
        if proc.poll() is not None:
            err = (proc.stderr.read() or "").strip().splitlines()
            tail = "\n  ".join(err[-6:]) if err else "(无输出)"
            print(f"\n✗ Aria 桥接启动失败：\n  {tail}\n", file=sys.stderr)
            return None, None
        if _port_open(args.aria_port):
            print(f"  ✓ Aria 桥接就绪 {url}")
            return proc, url
        time.sleep(0.1)
    print("\n✗ Aria 桥接 60 秒内没就绪", file=sys.stderr)
    _stop_aria_bridge(proc)
    return None, None


def _stop_aria_bridge(proc) -> None:
    """用 SIGINT 收尾，让桥接自己跑完 stop_streaming —— 强杀会在眼镜上留下
    占着的流式会话，下次启动就报 (940)。"""
    if proc is None or proc.poll() is not None:
        return
    print("  停止 Aria 桥接 ...")
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        print("  ⚠ 桥接没能正常退出。若下次报 (940)，先跑："
              "  ~/aria_env/bin/aria streaming stop")


def _check_camera(cap, tries: int = 40) -> bool:
    """真读几帧再说。macOS 没给权限时 VideoCapture 会「打开成功」但一帧读不到。"""
    for _ in range(tries):
        ok, frame = cap.read()
        if ok and frame is not None:
            return True
        time.sleep(0.05)
    return False


def main(argv=None) -> int:
    if _MISSING:
        _wrong_env(_MISSING)
    args = build_parser().parse_args(argv)
    if args.list_devices:
        return _list_devices()
    if args.mic_test:
        return _mic_test(device=args.mic, threshold=args.vad_threshold)
    cfg = load_config(args.config)
    if args.model:
        cfg.model.siglip = args.model
    if args.negatives:
        cfg.negatives.mode = args.negatives
    if args.fps:
        cfg.gate.fps = args.fps
    if args.no_preset:
        cfg.queries = []
    elif args.only:
        cfg.queries = [q for q in cfg.queries if q.id in set(args.only)]

    print(f"加载 SigLIP：{cfg.model.siglip} ...", flush=True)
    enc = SiglipEncoder(cfg.model)
    print(f"  device={enc.device} dtype={enc.dtype} 加载耗时={enc.load_s:.1f}s")

    gate = VisualGate(cfg, enc)
    gate.on_probe_done = _on_probe_done
    if gate.queries:
        print(f"  启用 {len(gate.queries)} 个视觉 query")
        for q in gate.queries:
            print(f"    [{q.trigger_type.value:8s}] {q.id:12s} {q.text}")
    else:
        print("  当前没有任何 query —— 对着麦克风说一句就能注册")

    mic = None
    stt = None
    if not args.no_audio:
        stt = Transcriber(args.asr_model, language=args.language)
        lang = args.language or "自动判别"
        mode = "原文" if args.no_translate else "原文 + 英译（query 用英文）"
        print(f"  ASR 后端：{stt.backend}  语种={lang}  输出={mode}")
        if not args.no_translate and not stt.can_translate:
            print("  ⚠ 这个 ASR 模型不支持翻译（turbo 系列砍掉了 translate 任务）。")
            print("    说中文的话 query 会是中文，SigLIP 匹配不上。")
            print("    换成 --asr-model mlx-community/whisper-small-mlx")
        if not stt.available:
            print("  ⚠ 没有可用的 ASR 后端，语音通路关闭（视觉照常）")
            stt = None
        else:
            mic = MicStream(RmsVad(threshold=args.vad_threshold), device=args.mic)

    aria_proc = None
    if args.camera is None:
        source = args.source or (_choose_source() if sys.stdin.isatty() else "mac")
        if source == "aria":
            aria_proc, url = _start_aria_bridge(args)
            if url is None:
                return 2
            args.camera = url
        else:
            args.camera = 0

    source_arg = int(args.camera) if str(args.camera).isdigit() else args.camera
    cap = cv2.VideoCapture(source_arg)
    if not cap.isOpened() or not _check_camera(cap):
        cap.release()
        _stop_aria_bridge(aria_proc)
        print(
            f"\n✗ 摄像头 {args.camera} 打不开，或者打开了但一帧都读不到。\n"
            f"  macOS 上最常见的原因是没授权：\n"
            f"    系统设置 → 隐私与安全性 → 摄像头 → 勾上你运行命令的那个程序\n"
            f"    （终端 Terminal / iTerm / VSCode，不是 Python）\n"
            f"  授权后要完全退出那个程序再重开，改完不重启不生效。\n"
            f"  用 --list-devices 看有哪些可用设备。\n",
            file=sys.stderr,
        )
        return 2
    print(f"  ✓ 画面来源 {args.camera} 正常出帧")

    jsonl_fp = open(args.jsonl, "w", encoding="utf-8") if args.jsonl else None
    sink = ConsoleSink(jsonl=jsonl_fp)
    slot = _Slot()
    lock = threading.Lock()

    def vision_worker() -> None:
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
            with lock:
                events = gate.step(frame, t)
            sink.emit(events)

    def audio_worker() -> None:
        n_new = 0
        seen_reject = 0
        while not slot.closed:
            try:
                utt = mic.out.get(timeout=0.2)
            except Exception:
                # 有声音但被判掉了，说一声，别让用户对着空气喊
                if mic.vad.rejected > seen_reject:
                    seen_reject = mic.vad.rejected
                    why = mic.vad.last_reject or ""
                    # 「太短」绝大多数是东西碰到桌面那一声，过滤器正常工作，不用报
                    if not why.startswith("太短") and not why.startswith("整段"):
                        print(f"\n\033[2m   （检测到声音但没收：{why}）\033[0m")
                continue
            native, english = stt.hear(utt.audio, want_english=not args.no_translate)
            if is_hallucination(native) and is_hallucination(english):
                print(f"\n\033[2m   （识别为静音幻觉，已丢弃：{native[:30]}）\033[0m")
                continue
            dur = utt.t_end - utt.t_start
            print(f"\n\033[1m🎙  听到\033[0m（{dur:.1f}s, rms={utt.rms:.3f}）：{native}")
            if english and english != native:
                print(f"    \033[2m英译：\033[0m{english}")

            cmd = voice_command(native, english)
            if cmd == "clear":
                with lock:
                    n = gate.clear_queries()
                print(f"    ✓ 已清空 {n} 条监测任务\n")
                continue
            if cmd == "list":
                with lock:
                    qs = list(gate.queries)
                print(f"    当前在监测 {len(qs)} 条：")
                for q in qs:
                    st = gate._state.get(q.id)
                    thr = (f"  min_raw={st.min_raw:.4f} delta_on={st.delta_on:.4f}"
                           if st is not None and st.min_raw is not None else "")
                    print(f"      [{q.trigger_type.value:8s}] {q.id:12s} {q.text}{thr}")
                print()
                continue

            kind = classify_intent(native, english)
            print(f"    判为 {kind}")
            if kind == "STANDING":
                visual = to_visual_prompt(english or native)
                with lock:
                    dup = gate.find_by_text(visual)
                if dup is not None:
                    print(f"    · 已经在监测同一件事了 [{dup.id}]：{dup.text}\n")
                    continue
                n_new += 1
                spec = QuerySpec(
                    id=f"sq_live{n_new}",
                    text=visual,
                    trigger_type=TriggerType.STANDING,
                    urgency=Urgency.NORMAL,
                )
                with lock:
                    gate.register_standing(spec)
                st = gate._state.get(spec.id)
                print(f"    ✓ 已注册 [{spec.id}]，视觉门开始监测")
                print(f"      匹配用描述：\033[1m{visual}\033[0m")
                if st is not None and st.min_raw is not None:
                    print(f"      本条阈值：min_raw={st.min_raw:.4f} delta_on={st.delta_on:.4f}")
                print()
            else:
                with lock:
                    seq = gate.next_seq()
                sink.emit([emit_instant(seq, utt.t_start, english or native, audio_rms=utt.rms)])

    threads = [threading.Thread(target=vision_worker, daemon=True)]
    if mic is not None:
        mic.start()
        print(f"  ✓ 麦克风 [{mic.device_name}] 已打开")
        print(f"    正在测环境底噪（{mic.vad.calibrate_s:g} 秒，请保持安静）...", end="", flush=True)
        t_cal = time.perf_counter()
        while not mic.vad.calibrated and time.perf_counter() - t_cal < 6.0:
            time.sleep(0.1)
        if mic.vad.calibrated:
            print(f" 底噪={mic.vad.noise_floor:.4f}  触发阈值={mic.vad.threshold:.4f}")
        else:
            print(" 超时（麦克风没出数据？用 --list-devices 检查）")
        threads.append(threading.Thread(target=audio_worker, daemon=True))
    for th in threads:
        th.start()

    _on, _off, _mr = gate.thresholds()
    _sm = _score_mode_label(cfg, gate)
    print(f"\n门采样率 {cfg.gate.fps} FPS，打分={_sm}，on={_on} min_raw={_mr}")
    print("对着麦克风说话即可：")
    print("  · 「如果我拿起手机就提醒我」   → 注册一条持续监测")
    print("  · 「这是什么」                 → 即时提问，直接出标签")
    print("  · 「我在监测什么」             → 列出当前任务")
    print("  · 「全部取消」                 → 清空所有任务")
    print("按 q 或 Ctrl-C 退出。\n")

    t0 = time.perf_counter()
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            t = time.perf_counter() - t0
            slot.put(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t)
            if not args.no_display:
                cv2.imshow("WHEN live — 按 q 退出", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.005)
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        slot.closed = True
        for th in threads:
            th.join(timeout=2.0)
        if mic is not None:
            mic.stop()
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        if jsonl_fp:
            jsonl_fp.close()
        _stop_aria_bridge(aria_proc)

    print(f"\n{'='*60}")
    print(f"结束  {sink.summary()}")
    print(f"SigLIP 编码耗时  {enc.timing_report()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
