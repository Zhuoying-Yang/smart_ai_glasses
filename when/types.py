"""WHEN 层的对外契约：一个事件流。

下游（WHICH router / 记忆模块）只依赖这里定义的 WhenEvent，
不依赖 gate 内部怎么算分。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional


class Route(str, Enum):
    TRIGGER = "TRIGGER"
    SILENT = "SILENT"


class TriggerType(str, Enum):
    INSTANT = "INSTANT"    # 用户即刻提问，WHEN 不做视觉判断，直接放行
    STANDING = "STANDING"  # 用户注册的持续监测任务
    ALERT = "ALERT"        # 系统预置的预警任务


class QueryOrigin(str, Enum):
    USER_SPEECH = "user_speech"      # 麦克风 -> ASR
    USER_STANDING = "user_standing"  # 用户注册的 standing query
    SYSTEM_PRESET = "system_preset"  # 内置预警清单


class Urgency(str, Enum):
    HIGH = "high"
    NORMAL = "normal"


@dataclass
class QueryRef:
    text: str
    origin: QueryOrigin
    query_id: Optional[str] = None


@dataclass
class Evidence:
    """触发依据。frame_idx 直接是 WHICH 的 HOW MUCH 输入——送几帧下去。"""

    window: List[float] = field(default_factory=list)     # [t_start, t_end] 秒
    frame_idx: List[int] = field(default_factory=list)
    novelty: Optional[float] = None      # 1 - cos(f_t, f_{t-1})，Dispider 的切段信号
    roi_box: Optional[List[int]] = None  # [x, y, w, h]，ROI 选中的框；也是给 WHICH 的「该看哪块」
    audio_rms: Optional[float] = None


@dataclass
class WhenEvent:
    seq: int
    t_emit: float                 # 流内时间戳（秒），流起点为 0
    route: Route
    trigger_type: TriggerType
    query: QueryRef
    score: float                  # 连续分，SILENT 时也给——别重蹈 Dispider 阈值写死的覆辙
    threshold: float
    evidence: Evidence = field(default_factory=Evidence)
    urgency: Urgency = Urgency.NORMAL
    cooldown_until: Optional[float] = None
    raw_score: Optional[float] = None    # 未经 EMA 的瞬时分，调参时有用
    baseline: Optional[float] = None     # 该 query 自己的长期基线（relative 模式）
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    t_wall: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=str)

    def is_trigger(self) -> bool:
        return self.route is Route.TRIGGER


__all__ = [
    "Route",
    "TriggerType",
    "QueryOrigin",
    "Urgency",
    "QueryRef",
    "Evidence",
    "WhenEvent",
]
