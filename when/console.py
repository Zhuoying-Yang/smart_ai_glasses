"""终端输出。TRIGGER 显眼，SILENT 压成一行状态条。"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, TextIO

from .types import Route, TriggerType, WhenEvent

_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_COLOR = {
    TriggerType.INSTANT: "\033[36m",   # 青
    TriggerType.STANDING: "\033[33m",  # 黄
    TriggerType.ALERT: "\033[31m",     # 红
}


def _tty() -> bool:
    return sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}" if _tty() else text


def _bar(v: float, width: int = 10) -> str:
    n = max(0, min(width, int(round(v * width))))
    return "█" * n + "·" * (width - n)


class ConsoleSink:
    def __init__(self, verbose: bool = False, jsonl: Optional[TextIO] = None):
        self.verbose = verbose
        self.jsonl = jsonl
        self.n_trigger = 0
        self.n_event = 0

    def emit(self, events: List[WhenEvent]) -> None:
        if not events:
            return
        self.n_event += len(events)
        if self.jsonl:
            for e in events:
                self.jsonl.write(e.to_json() + "\n")
            self.jsonl.flush()

        triggers = [e for e in events if e.route is Route.TRIGGER]
        silents = [e for e in events if e.route is Route.SILENT]

        # 状态条：一行看完所有 query 当前分数
        if silents or triggers:
            t = events[0].t_emit
            parts: List[str] = []
            for e in sorted(events, key=lambda x: (x.trigger_type.value, x.query.query_id or "")):
                qid = e.query.query_id or "instant"
                # 条形显示绝对相似度，数字显示相对基线的抬升量（触发依据）
                bar = _bar(e.raw_score if e.raw_score is not None else e.score, 8)
                parts.append(f"{qid}:{bar}{e.score:+.2f}")
            line = f"{_DIM}[{t:7.2f}s]{_RESET} " + "  ".join(parts)
            print(line if _tty() else line.replace(_DIM, "").replace(_RESET, ""))

        for e in triggers:
            self.n_trigger += 1
            col = _COLOR.get(e.trigger_type, "")
            head = _c(f"▶ TRIGGER {e.trigger_type.value}", col + _BOLD)
            nov = f" nov={e.evidence.novelty:.3f}" if e.evidence.novelty is not None else ""
            raw = f"{e.raw_score:.3f}" if e.raw_score is not None else "-"
            base = f"{e.baseline:.3f}" if e.baseline is not None else "-"
            frames = f"  frames={e.evidence.frame_idx}" if e.evidence.frame_idx else ""
            print(
                f"[{e.t_emit:7.2f}s] {head}  {_c(e.query.query_id or '-', _BOLD)}  "
                f"lift={e.score:+.3f} thr={e.threshold:.2f} (raw={raw} base={base})"
                f"{nov}  urgency={e.urgency.value}{frames}"
            )
            print(f"           {_DIM}query:{_RESET} {e.query.text}")

    def summary(self) -> str:
        return f"events={self.n_event}  triggers={self.n_trigger}"


__all__ = ["ConsoleSink"]
