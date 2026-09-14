"""语音通路：RMS VAD 切句 -> 整段转文字 -> 判断是 INSTANT 还是 STANDING。

刻意不做流式 ASR。用户说完一句才转一次，一句 2-5 秒在 M4 上远小于 1 秒，
体感就是说完立刻出字，复杂度却低一个数量级。
"""

from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

SAMPLE_RATE = 16000

# 命中这些说法 -> 当成 STANDING（注册一个持续监测任务），否则当 INSTANT（即刻提问）
_STANDING_PATTERNS = [
    # 中文
    r"提醒我", r"提醒一下", r"帮我盯", r"帮我看着", r"帮我留意", r"留意",
    r"监测", r"监控", r"注意看", r"盯着",
    r"一旦", r"如果.*就", r"当.*(时候|的时候|时)", r"只要.*就", r".*的时候告诉我",
    # 英文。ASR 常把 warn 听成 warm、warning，所以放宽到词干
    r"\bwar[nm]\b", r"\bwarning\b", r"\balert\b", r"\bremind\b", r"\bnotify\b",
    r"\blet me know\b", r"\btell me (when|if|once)\b",
    r"\bwatch (for|out)\b", r"\bkeep an eye\b", r"\bmonitor\b",
    r"\bwhenever\b", r"\bas soon as\b", r"\bif i\b", r"\bwhen (someone|somebody|i|you|the)\b",
]

_STANDING_RE = re.compile("|".join(_STANDING_PATTERNS), re.IGNORECASE)


# 运行中用嘴管理 query
CLEAR_RE = re.compile(
    r"清空|全部取消|取消所有|都取消|别监测了|停止监测|忘掉|重新开始"
    r"|\bclear (all|everything)\b|\bcancel (all|everything)\b"
    r"|\bforget (all|everything|it)\b|\bstop watching\b|\breset\b",
    re.IGNORECASE,
)
LIST_RE = re.compile(
    r"列出|有哪些|在监测什么|当前.*(任务|监测)|\blist\b|\bwhat are you watching\b",
    re.IGNORECASE,
)


def voice_command(*texts: str) -> Optional[str]:
    """识别「清空」「列出」这类管理指令。返回 'clear' / 'list' / None。"""
    for t in texts:
        if not t:
            continue
        if CLEAR_RE.search(t):
            return "clear"
        if LIST_RE.search(t):
            return "list"
    return None


# Whisper 在静音上会反复吐这几句。它们不是识别结果，是幻觉。
_HALLUCINATION = re.compile(
    r"^\W*("
    r"you|thank you|thanks|bye|okay|ok|uh|um|hmm|so|yeah"
    r"|thank you for watching|thanks for watching|please subscribe"
    r"|subscribe to my channel|see you next time"
    r"|字幕由.*提供|请不吝点赞|訂閱|谢谢观看|感谢观看|下集再见"
    r")\W*$",
    re.IGNORECASE,
)


def is_hallucination(text: str) -> bool:
    """静音幻觉：要么命中黑名单，要么整句是同一个字符刷屏。"""
    t = (text or "").strip()
    if not t:
        return True
    if _HALLUCINATION.match(t):
        return True
    letters = [c for c in t if not c.isspace()]
    if len(letters) >= 12 and len(set(letters)) <= 2:
        return True
    return False


def _is_mostly_ascii(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    return sum(c.isascii() for c in letters) / len(letters) > 0.9


def classify_intent(text: str, *alts: str) -> str:
    """任一说法命中 STANDING 模式就算 STANDING。

    传原文和英译两份进来，中英规则互为补充，ASR 转错一个也还有另一个。
    """
    for t in (text, *alts):
        if t and _STANDING_RE.search(t):
            return "STANDING"
    return "INSTANT"


def _classify_one(text: str) -> str:
    return "STANDING" if _STANDING_RE.search(text or "") else "INSTANT"


# 指令外壳 -> 剥掉，只留画面描述。SigLIP 匹配的是画面里有什么，
# 不是「提醒我」这种祈使语气。
_STRIP_LEAD = [
    r"^\s*(please\s+)?let\s+me\s+know\s*(if|when|once|that)?\s*",
    r"^\s*(please\s+)?(warn|alert|notify|remind|tell|let)\s+me\s*(if|when|once|as soon as|that)?\s*",
    r"^\s*(watch|look)\s+(out\s+)?(for|when)\s*",
    r"^\s*keep an eye\s+(out\s+)?(for|on)\s*",
    r"^\s*monitor\s+(for)?\s*",
    r"^\s*(如果|要是|当|一旦|只要|假如)\s*",
    r"^\s*(when|if|once|whenever)\s+(i|you|someone|somebody|the|there)\s+",
    r"^\s*(帮我)?(盯着|看着|留意|监测|监控)\s*",
]
_STRIP_TAIL = [
    r"\s*(的时候|时候|时)?\s*(就)?\s*(提醒|告诉|通知)我\s*[。.！!]?\s*$",
    r"\s*[,，]?\s*(please\s+)?(warn|alert|tell|notify|remind|let)\s+me(\s+know)?\s*[.!]?\s*$",
]
_PRONOUN = [
    (r"\bmy\b", "the"), (r"\byour\b", "the"),
    (r"^\s*i\s+", ""), (r"^\s*我\s*(把|在|要)?\s*", ""),
]


# 动作短语 -> 画面描述。
# SigLIP 匹配的是「画面里有什么」，不是「发生了什么动作」。
# "take the phone" 对它几乎等于 "get the cup" —— 都是一只手伸进画面，
# 所以多条 query 会一起动。换成名词短语才能拉开区分度。
_ACTION_TEMPLATES = [
    (r"^(?:take|get|grab|pick\s*up|hold|lift|reach\s+for|use|touch)\s+(?:the|a|an|my|his|her|their)?\s*(.+)$",
     "a hand holding {}"),
    (r"^(?:put\s+on|wear|wearing)\s+(?:the|a|an|my|his|her|their)?\s*(.+)$",
     "a person wearing {}"),
    (r"^(?:open|opening)\s+(?:the|a|an|my)?\s*(.+)$", "a person opening {}"),
    (r"^(?:close|closing|shut)\s+(?:the|a|an|my)?\s*(.+)$", "a person closing {}"),
    (r"^(?:drink|drinking|sip)\s+(?:from\s+)?(?:the|a|an|my)?\s*(.+)$",
     "a person drinking from {}"),
    (r"^(?:someone|somebody|a\s+person)\s+(?:comes?|enters?|walks?)\s+in.*$",
     "a person entering the room"),
]
_TRAILING_JUNK = re.compile(
    r"[,，]?\s*(it|he|she|they)\s+(will|would|should|can)\s*$|[,，]\s*$", re.IGNORECASE
)


def _to_visual_phrase(text: str) -> str:
    """把动作短语套成名词短语。套不上就原样返回。"""
    t = _TRAILING_JUNK.sub("", text).strip()
    for pat, tpl in _ACTION_TEMPLATES:
        m = re.match(pat, t, flags=re.IGNORECASE)
        if m:
            obj = (m.groups()[-1] if m.groups() else "").strip(" .,!?")
            if not obj:
                return tpl.replace(" {}", "")
            plural = re.search(r"[^s]s$|[^aeiou]ies$", obj, flags=re.IGNORECASE)
            if not plural and not re.match(r"^(a|an|the|some)\s", obj, flags=re.IGNORECASE):
                obj = ("an " if obj[:1].lower() in "aeiou" else "a ") + obj
            return tpl.format(obj)
    return t


def to_visual_prompt(text: str) -> str:
    """把口头指令剥成一句画面描述。规则式，以后可以换成一次小模型改写。"""
    out = (text or "").strip()
    for pat in _STRIP_LEAD:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    for pat in _STRIP_TAIL:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    for pat, rep in _PRONOUN:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    out = out.strip(" ,.!?，。！？")
    out = _to_visual_phrase(out)
    return out or (text or "").strip()


@dataclass
class Utterance:
    t_start: float
    t_end: float
    audio: np.ndarray
    rms: float


class RmsVad:
    """能量阈值 + 挂起时间 + 环境噪声自标定。

    上一版最大的毛病：阈值写死 0.012，环境稍微吵一点就被噪声触发，
    然后把半秒近静音送给 Whisper —— Whisper 在静音上必然幻觉，
    吐出 "you" / "Thank you for watching!" / 一串僧伽罗字母。
    """

    def __init__(
        self,
        threshold: float = 0.006,   # 只是地板值，实际阈值由环境底噪决定
        min_speech_s: float = 0.45,     # 短于这个不送去识别，Whisper 会瞎编
        hangover_s: float = 0.7,
        max_utt_s: float = 12.0,
        onset_blocks: int = 2,          # 连续几块都够响才算起音，滤掉瞬时噪声
        snr_ratio: float = 1.6,         # 整句平均能量要比环境底噪高这么多倍
        calibrate_s: float = 1.5,       # 开头多久用来量环境底噪
    ):
        self.threshold = threshold
        self.min_speech_s = min_speech_s
        self.hangover_s = hangover_s
        self.max_utt_s = max_utt_s
        self.onset_blocks = onset_blocks
        self.snr_ratio = snr_ratio
        self.calibrate_s = calibrate_s

        self.noise_floor: Optional[float] = None
        self._cal: List[float] = []
        self._buf: List[np.ndarray] = []
        self._rms: List[float] = []      # 每块的能量，用来剪掉尾部静音
        self._hot = 0
        self._speaking = False
        self._t_start = 0.0
        self._last_voice = 0.0
        self.last_rms = 0.0
        self.peak_rms = 0.0
        self.rejected = 0
        self.accepted = 0
        self.last_reject: Optional[str] = None

    @property
    def calibrated(self) -> bool:
        return self.noise_floor is not None

    def push(self, chunk: np.ndarray, t: float) -> Optional[Utterance]:
        rms = float(np.sqrt(np.mean(np.square(chunk))) + 1e-12)
        self.last_rms = rms
        self.peak_rms = max(self.peak_rms, rms)

        # 开头一段只听不判，用来量环境底噪
        if self.noise_floor is None:
            self._cal.append(rms)
            if t >= self.calibrate_s and self._cal:
                floor = float(np.percentile(self._cal, 75))
                self.noise_floor = floor
                # 宁可松一点让噪声偶尔进来（后面有幻觉黑名单兜底），
                # 也不要严到把人说的话挡在外面 —— 挡掉了用户是完全无感的
                self.threshold = max(self.threshold, floor * 2.0)
            return None

        voiced = rms > self.threshold
        self._hot = self._hot + 1 if voiced else 0

        if not self._speaking and self._hot >= self.onset_blocks:
            self._speaking = True
            self._t_start = t
            self._last_voice = t
            self._buf = []
            self._rms = []

        if not self._speaking:
            return None

        self._buf.append(chunk)
        self._rms.append(rms)
        if voiced:
            self._last_voice = t
        if t - self._t_start < self.max_utt_s and (t - self._last_voice) < self.hangover_s:
            return None

        blocks, rmss = self._buf, self._rms
        self._speaking = False
        self._hot = 0
        self._buf = []
        self._rms = []

        # 前后静音都要剪掉。
        # 尾部：挂起期那 0.7 秒会稀释均值，也让 Whisper 多听半秒噪声。
        # 前部：噪声先触发「开始说话」、真人声落在中段时，前面一大截静音
        #       能把均值压到底噪以下，真话反而被判成「太轻」。
        voiced_idx = [i for i, r in enumerate(rmss) if r > self.threshold]
        if not voiced_idx:
            self.rejected += 1
            self.last_reject = "整段都没有超过阈值的块"
            return None
        lo = max(0, voiced_idx[0] - 1)            # 前后各多留一块，别切掉起音和尾音
        hi = min(len(blocks), voiced_idx[-1] + 2)
        keep = blocks[lo:hi]
        audio = np.concatenate(keep)
        block_s = len(blocks[0]) / SAMPLE_RATE if blocks else 0.05
        dur = len(keep) * block_s
        mean_rms = float(np.sqrt(np.mean(np.square(audio))) + 1e-12)

        if dur < self.min_speech_s:
            self.rejected += 1
            self.last_reject = f"太短 有效{dur:.2f}s < {self.min_speech_s:g}s"
            return None
        if mean_rms < self.noise_floor * self.snr_ratio:
            self.rejected += 1
            self.last_reject = (
                f"太轻 有效段均值{mean_rms:.4f} < 底噪{self.noise_floor:.4f}×{self.snr_ratio:g}"
            )
            return None
        self.last_reject = None
        self.accepted += 1
        return Utterance(self._t_start, self._t_start + dur, audio, mean_rms)


class Transcriber:
    """优先 mlx-whisper（Apple 原生 Metal），装不上自动退到 faster-whisper。

    Whisper 自带两个任务：
      transcribe -> 原文（你说中文就出中文）
      translate  -> 一律输出英文
    正好满足「说中文，但 query 用英文喂 SigLIP」。
    """

    def __init__(
        self,
        model: str = "mlx-community/whisper-small-mlx",
        language: Optional[str] = None,
    ):
        self.language = language          # None = 自动判别语种
        # large-v3-turbo 训练时砍掉了 translate 任务，喂它 task="translate" 会原样返回中文。
        # small 反而又快又能翻（实测 RTF 0.14-0.20x vs turbo 的 0.68-2.24x）。
        self.can_translate = "turbo" not in model.lower()
        self.backend = None
        self._fn: Optional[Callable[[np.ndarray, str], str]] = None

        try:
            import mlx_whisper  # noqa

            def _mlx(audio: np.ndarray, task: str) -> str:
                out = mlx_whisper.transcribe(
                    audio, path_or_hf_repo=model, language=self.language, task=task
                )
                return (out.get("text") or "").strip()

            self._fn, self.backend = _mlx, f"mlx-whisper:{model}"
            return
        except Exception:
            pass

        try:
            from faster_whisper import WhisperModel

            m = WhisperModel("small", device="cpu", compute_type="int8")

            def _fw(audio: np.ndarray, task: str) -> str:
                segs, _ = m.transcribe(audio, language=self.language, task=task)
                return " ".join(s.text for s in segs).strip()

            self._fn, self.backend = _fw, "faster-whisper:small(cpu-int8)"
            return
        except Exception:
            pass

        self.backend = "none"

    @property
    def available(self) -> bool:
        return self._fn is not None

    def transcribe(self, audio: np.ndarray, task: str = "transcribe") -> str:
        if self._fn is None:
            return ""
        return self._fn(audio.astype(np.float32), task)

    def hear(self, audio: np.ndarray, want_english: bool = True):
        """返回 (原文, 英文)。

        原文给你看，英文喂 SigLIP —— SigLIP 的文本塔是英文训练的，
        直接喂中文匹配不上。
        """
        a = audio.astype(np.float32)
        native = self.transcribe(a, "transcribe")
        if not want_english or not native:
            return native, native
        if _is_mostly_ascii(native):
            return native, native            # 本来就是英文，省一次前向
        if not self.can_translate:
            return native, native            # 这个模型不会翻，别白跑一次
        return native, self.transcribe(a, "translate")


class MicStream:
    """后台采麦克风，VAD 切句后把 Utterance 丢进队列。"""

    def __init__(
        self,
        vad: Optional[RmsVad] = None,
        block_s: float = 0.05,
        device: Optional[int] = None,
    ):
        self.vad = vad or RmsVad()
        self.block = int(SAMPLE_RATE * block_s)
        self.device = device
        self.out: "queue.Queue[Utterance]" = queue.Queue()
        self._stream = None
        self._t0 = 0.0
        self._n = 0

    @property
    def device_name(self) -> str:
        import sounddevice as sd

        idx = self.device if self.device is not None else sd.default.device[0]
        try:
            return str(sd.query_devices(idx)["name"])
        except Exception:
            return f"device {idx}"

    def start(self) -> None:
        import sounddevice as sd

        self._t0 = time.perf_counter()

        def cb(indata, frames, time_info, status):  # noqa: ANN001
            # 必须 copy()。PortAudio 每次回调复用同一块输入缓冲，
            # np.asarray 对 float32 只返回视图 —— 攒进 _buf 的所有块都会
            # 指向同一块内存，拼出来的是最后一次回调的内容（通常是静音）。
            # 这个 bug 会让 rms 读数正常但录到的波形全是静音。
            chunk = np.array(indata[:, 0], dtype=np.float32, copy=True)
            self._n += frames
            t = self._n / SAMPLE_RATE
            utt = self.vad.push(chunk, t)
            if utt is not None:
                self.out.put(utt)

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=self.block,
            callback=cb,
            device=self.device,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


__all__ = [
    "SAMPLE_RATE",
    "to_visual_prompt",
    "voice_command",
    "is_hallucination",
    "RmsVad",
    "Utterance",
    "Transcriber",
    "MicStream",
    "classify_intent",
]
