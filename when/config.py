"""加载 queries.yaml。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

from .types import TriggerType, Urgency

DEFAULT_CONFIG = Path(__file__).with_name("queries.yaml")


@dataclass
class QuerySpec:
    id: str
    text: str
    trigger_type: TriggerType
    urgency: Urgency = Urgency.NORMAL


@dataclass
class ModelCfg:
    siglip: str = "google/siglip-so400m-patch14-384"
    device: str = "auto"
    dtype: str = "float16"


@dataclass
class GateCfg:
    fps: float = 2.0
    softmax_scale: float = 30.0
    cooldown_s: float = 20.0
    emit_silent: bool = True

    # relative（默认）：看分数相对自身基线抬升了多少 —— 判断「事件发生了」
    # absolute：看分数绝对值 —— 判断「画面里有这个东西」
    mode: str = "relative"

    ema_alpha: float = 0.35        # 快 EMA：当前状态
    baseline_alpha: float = 0.03   # 慢 EMA：该 query 自己的长期基线
    warmup_s: float = 3.0          # 慢 EMA 稳定前不触发
    snap_baseline_on_fire: bool = True   # 触发后把基线拉到当前值，避免同一事件反复重报
    latch_until_absent: bool = True      # 触发后锁住，直到目标从画面消失才重新武装

    delta_on: float = 0.10         # relative 模式的滞回阈值（快 - 慢）
    delta_off: float = 0.04
    min_raw: float = 0.55          # 绝对下限：raw 没到这个值一律不触发

    # negatives 留空时自动退回纯余弦模式（零配置，换环境直接能用）。
    # 余弦的数值范围只有 0~0.2，和 softmax 的 0~1 差一个量级，所以另给一套阈值。
    cos_delta_on: float = 0.003
    cos_min_raw: float = 0.155
    tau_on: float = 0.55           # absolute 模式的滞回阈值
    tau_off: float = 0.40


@dataclass
class AutoNegCfg:
    """自动负样本：开头探测几秒，从词表里挑出最贴合当前环境的几条。"""

    probe_seconds: float = 4.0
    top_k: int = 6
    vocab: str = "builtin"          # 'builtin' 或词表文件路径
    exclude_similar: float = 0.90   # 与任一 query 文本相似度超过此值的词条不选

    # 只按分数取前 K 会让同一语义簇的条目一起挤进来
    # （实测一次选出的 6 条里 4 条都是「桌面电脑周边」，只覆盖 2 个簇）。
    #
    # max_per_category 是主力：词表分场景/物体/人/画质四类，每类限额。
    # 实测这个比基于嵌入的 MMR 有效得多 —— 句子嵌入的两两相似度都在
    # 0.69-0.84 之间，方差太小，MMR 推不动排序。
    max_per_category: int = 2
    # MMR（嵌入层面的多样性）默认关闭：在眼镜/书桌/厨房三个场景上实测，
    # 它只在一个场景改变了结果，而且只是换了两条的顺序、集合完全一样。
    # 句子嵌入的两两相似度都挤在 0.69-0.84，方差太小，推不动排序。留着旋钮但不开。
    diversity: float = 0.0
    reprobe_on_scene_change: bool = True
    scene_change_novelty: float = 0.35

    # 探测期顺便标定 min_raw。自动负样本会把所有分数整体拉低，
    # 固定阈值会立刻失准，所以两者必须一起自适应。
    # 阈值按探测期分布的**尺度**定（μ + kσ），不能用绝对值：
    # 负样本越贴合环境，query 的概率量纲越小（厨房 0.088 vs 书桌 0.73，差一个数量级）
    calibrate_min_raw: bool = True
    sigma_min_raw: float = 3.0      # min_raw = μ + 3σ（只是下界之一）
    sigma_delta_on: float = 4.0     # delta_on = 4σ（只是下界之一）
    delta_on_floor: float = 0.004
    min_raw_floor: float = 0.02
    min_raw_cap: float = 0.75       # 上限，防止探测期恰好有目标物体把阈值抬太高

    # 探测期是静止场景，σ 量的是「什么都没发生时的抖动」，
    # 但真正要区分的是「举起对的东西」和「举起错的东西」—— 尺度大一个数量级。
    # 所以再加一个有依据的锚点：softmax 有 K 条负样本时，纯随机的概率是 1/(1+K)，
    # 低于这个数说明模型根本没匹配上。实测(2026-09-14 live)：
    #   正确触发 raw∈[0.21, 0.31]，误触发 raw∈[0.06, 0.12]，K=6 时 1/(1+K)=0.143 正好落在空档里
    chance_multiplier: float = 1.2  # min_raw 至少要有 1.2 × 随机水平
    # 但随机水平是个固定值（K=6 时 0.143），而各场景的分数量纲差一个数量级：
    # 负样本越贴合环境，query 的概率被压得越低。实测厨房 auto 下所有 query 的
    # 全程峰值都在 0.017-0.094，固定的 0.171 是谁都够不到的墙，导致一次都不触发。
    # 所以给 chance 锚点加个天花板：不能超过「探测期统计量 × headroom」。
    chance_headroom: float = 3.0
    delta_ratio: float = 0.20       # delta_on 至少要有 min_raw 的 20%


@dataclass
class NegativesCfg:
    mode: str = "manual"            # off（纯余弦·零配置）| manual | auto
    manual: List[str] = field(default_factory=list)
    auto: AutoNegCfg = field(default_factory=AutoNegCfg)


@dataclass
class RoiCfg:
    """条件式网格裁剪。默认关闭。

    只在「全图分数接近 min_raw 但没够到」时才裁——绝大多数帧不触发，
    所以平均开销几乎不变。这是 WHEN 内部的一层小 cascade。
    """

    enabled: bool = False
    grid: int = 3              # grid×grid 个候选框
    frac: float = 0.5          # 每个框占全图的边长比例
    margin: float = 0.20       # 全图分在 [min_raw-margin, min_raw) 区间内才裁
    max_crops: int = 4         # 每次最多裁几块（0 = 不限）


@dataclass
class WhenConfig:
    model: ModelCfg = field(default_factory=ModelCfg)
    gate: GateCfg = field(default_factory=GateCfg)
    roi: RoiCfg = field(default_factory=RoiCfg)
    negatives: NegativesCfg = field(default_factory=NegativesCfg)
    queries: List[QuerySpec] = field(default_factory=list)

    @property
    def visual_queries(self) -> List[QuerySpec]:
        """走视觉门的那些（STANDING + ALERT）。INSTANT 不经过视觉门。"""
        return [q for q in self.queries if q.trigger_type is not TriggerType.INSTANT]


def load_config(path: Path | str = DEFAULT_CONFIG) -> WhenConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    queries: List[QuerySpec] = []
    for section, ttype in (("standing", TriggerType.STANDING), ("alerts", TriggerType.ALERT)):
        for item in raw.get(section) or []:
            queries.append(
                QuerySpec(
                    id=item["id"],
                    text=item["text"],
                    trigger_type=ttype,
                    urgency=Urgency(item.get("urgency", "normal")),
                )
            )

    neg_raw = raw.get("negatives")
    if isinstance(neg_raw, list):           # 旧格式：直接就是一串句子
        negatives = NegativesCfg(mode="manual" if neg_raw else "off", manual=list(neg_raw))
    else:
        neg_raw = neg_raw or {}
        negatives = NegativesCfg(
            mode=neg_raw.get("mode", "manual"),
            manual=list(neg_raw.get("manual") or []),
            auto=AutoNegCfg(**(neg_raw.get("auto") or {})),
        )
        if negatives.mode == "manual" and not negatives.manual:
            negatives.mode = "off"

    seen = [q.id for q in queries]
    dupes = {i for i in seen if seen.count(i) > 1}
    if dupes:
        raise ValueError(
            f"queries.yaml 里有重复的 id: {sorted(dupes)}。"
            "每条 query 的 id 必须唯一，否则它们会共用同一份状态。"
        )

    return WhenConfig(
        model=ModelCfg(**(raw.get("model") or {})),
        gate=GateCfg(**(raw.get("gate") or {})),
        roi=RoiCfg(**(raw.get("roi") or {})),
        negatives=negatives,
        queries=queries,
    )


__all__ = [
    "WhenConfig", "QuerySpec", "ModelCfg", "GateCfg", "RoiCfg",
    "NegativesCfg", "AutoNegCfg", "load_config", "DEFAULT_CONFIG",
]
