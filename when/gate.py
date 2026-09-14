"""WHEN 的三条通路。

STANDING / ALERT  -> 视觉门（SigLIP zero-shot + EMA + 滞回 + 冷却）
INSTANT           -> 不经过视觉门，ASR 出文本就直接放行
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import QuerySpec, WhenConfig
from .vocab import load_vocab
from .encoder import SiglipEncoder
from .types import (
    Evidence,
    QueryOrigin,
    QueryRef,
    Route,
    TriggerType,
    Urgency,
    WhenEvent,
)

_MAX_PROBE_FRAMES = 60      # 探测帧缓冲上限，防止长时间没有 query 时无限攒

_ORIGIN = {
    TriggerType.STANDING: QueryOrigin.USER_STANDING,
    TriggerType.ALERT: QueryOrigin.SYSTEM_PRESET,
}


class _QueryState:
    """每个 query 一份独立状态。

    fast 是当前状态，slow 是这个 query 自己的长期基线。
    用 fast-slow 而不是 fast 本身做判断，才能区分
    「画面里一直有水瓶」（slow 也高，差值≈0）和
    「水瓶刚被挪到键盘旁」（fast 抬升，slow 还没跟上）。
    """

    def __init__(self) -> None:
        self.fast: Optional[float] = None
        self.slow: Optional[float] = None
        self.on: bool = False
        self.cooldown_until: float = -1.0
        self.n: int = 0
        # 每条 query 各标各的阈值。全局一个值是错的 ——
        # 不同 query 在同一场景下的分数量纲差很远，
        # 用第一条 query 标出来的门槛去卡后面注册的，后面那些永远够不到。
        self.min_raw: Optional[float] = None
        self.delta_on: Optional[float] = None

    def update(self, x: float, alpha_fast: float, alpha_slow: float) -> None:
        """bias correction：开头几十帧等价于算术平均，之后自动过渡成 EMA。

        没有它，慢基线要 1/alpha_slow ≈ 33 帧（2 FPS 下 16.5 秒）才爬到正确值，
        这段爬升期里 lift 完全不可信。有了它，第 2 帧的基线就已经是真实均值。
        """
        self.n += 1
        if self.fast is None:
            self.fast = self.slow = x
            return
        af = max(alpha_fast, 1.0 / self.n)
        ab = max(alpha_slow, 1.0 / self.n)
        self.fast = af * x + (1 - af) * self.fast
        self.slow = ab * x + (1 - ab) * self.slow

    @property
    def lift(self) -> float:
        """相对自身基线的抬升量。"""
        if self.fast is None or self.slow is None:
            return 0.0
        return self.fast - self.slow


def _grid_boxes(h: int, w: int, n: int, frac: float) -> List[Tuple[int, int, int, int]]:
    """n×n 个重叠候选框，每个边长占全图 frac。中心的框排在最前面（最常命中）。"""
    ch, cw = int(h * frac), int(w * frac)
    boxes = []
    for r in range(n):
        for c in range(n):
            y = min(r * (h - ch) // max(n - 1, 1), h - ch)
            x = min(c * (w - cw) // max(n - 1, 1), w - cw)
            boxes.append((x, y, cw, ch))
    mid = n // 2
    boxes.sort(key=lambda b: abs(b[0] + b[2] // 2 - w // 2) + abs(b[1] + b[3] // 2 - h // 2))
    return boxes


class VisualGate:
    def __init__(self, cfg: WhenConfig, encoder: SiglipEncoder):
        self.cfg = cfg
        self.enc = encoder
        self.queries: List[QuerySpec] = list(cfg.visual_queries)

        # query 文本是静态的，只编码一次
        self._q_emb = (
            self.enc.encode_texts([q.text for q in self.queries]) if self.queries else None
        )
        self._neg_emb: Optional[torch.Tensor] = None

        self._vocab: List[str] = []
        self._vocab_emb: Optional[torch.Tensor] = None
        self._probe: List[torch.Tensor] = []
        self._probe_t0: Optional[float] = None
        self._min_raw_override: Optional[float] = None
        self._delta_on_override: Optional[float] = None
        self.picked_negatives: List[str] = []
        self.on_probe_done = None            # 回调，给上层打印用

        mode = cfg.negatives.mode
        if mode == "manual" and cfg.negatives.manual:
            self._neg_emb = self.enc.encode_texts(cfg.negatives.manual)
        elif mode == "auto":
            self._vocab = load_vocab(cfg.negatives.auto.vocab)
            self._vocab_emb = self.enc.encode_texts(self._vocab)

        self._state: Dict[str, _QueryState] = {q.id: _QueryState() for q in self.queries}
        self._prev_frame_emb: Optional[torch.Tensor] = None
        self._frame_i = -1
        self._seq = 0

    @property
    def _text_emb(self) -> Optional[torch.Tensor]:
        if self._q_emb is None:
            return None
        if self._neg_emb is None:
            return self._q_emb
        return torch.cat([self._q_emb, self._neg_emb], dim=0)

    @property
    def _n_neg(self) -> int:
        return 0 if self._neg_emb is None else int(self._neg_emb.shape[0])

    @property
    def probing(self) -> bool:
        """auto 模式下还没定出负样本的那段时间。"""
        return self.cfg.negatives.mode == "auto" and self._neg_emb is None

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def find_by_text(self, text: str) -> Optional[QuerySpec]:
        key = (text or "").strip().lower()
        return next((q for q in self.queries if q.text.strip().lower() == key), None)

    def register_standing(self, spec: QuerySpec) -> None:
        """运行时新增一条 standing query（阶段 2 用户口头注册时调用）。"""
        self.queries.append(spec)
        self._state[spec.id] = _QueryState()
        new = self.enc.encode_texts([spec.text])
        self._q_emb = new if self._q_emb is None else torch.cat([self._q_emb, new], dim=0)
        # 环境已经探测过了，新 query 马上就能用那批帧标出自己的阈值
        if self._neg_emb is not None and self._probe:
            self._calibrate([len(self.queries) - 1])

    def _finish_probe(self) -> None:
        """探测期结束：从词表挑负样本，并顺便标定 min_raw。"""
        a = self.cfg.negatives.auto
        if not self._probe or self._vocab_emb is None:
            self._neg_emb = self.enc.encode_texts(["an ordinary scene"])
            return

        scene = torch.stack(self._probe).mean(0)
        scene = scene / scene.norm()
        sims = (self._vocab_emb @ scene).cpu().numpy()

        # 排除与任一 query 近乎重复的词条，否则等于把自己的 query 也否掉
        order = np.argsort(-sims)
        if self._q_emb is None:
            picked = [int(j) for j in order[: a.top_k]]
        else:
            qsim = (self._vocab_emb @ self._q_emb.T).cpu().numpy().max(axis=1)
            picked = [int(j) for j in order if qsim[j] < a.exclude_similar][: a.top_k]
        if not picked:
            picked = [int(j) for j in order[: a.top_k]]

        self.picked_negatives = [self._vocab[j] for j in picked]
        self._neg_emb = self._vocab_emb[picked].clone()

        if a.calibrate_min_raw:
            self._calibrate(range(len(self.queries)))

        if self.on_probe_done:
            self.on_probe_done(self)

    def _calibrate(self, indices) -> None:
        """用探测帧给指定的几条 query 各自标阈值。

        阈值必须按该 query 自己分数分布的**尺度**定，不能用绝对值：
        负样本越贴合环境，query 在 softmax 里的概率量纲越小
        （实测厨房峰值 0.088 vs 书桌 0.73，差一个数量级），
        而不同 query 之间的量纲差得同样远。
        """
        a = self.cfg.negatives.auto
        idx = [i for i in indices if 0 <= i < len(self.queries)]
        if not idx or not self._probe or self._neg_emb is None:
            return
        scores = np.stack([self._scores(e) for e in self._probe])   # [帧, query]
        # 随机水平：softmax 在 [该 query, *K 条负样本] 上打分，没信息时就是 1/(1+K)
        chance = 1.0 / (1.0 + max(self._n_neg, 1))
        for i in idx:
            col = scores[:, i]
            mu, sd = float(col.mean()), float(col.std()) + 1e-6
            st = self._state[self.queries[i].id]
            st.min_raw = float(
                min(
                    max(
                        mu + a.sigma_min_raw * sd,      # 高过静止期的抖动
                        chance * a.chance_multiplier,   # 也要高过随机水平
                        a.min_raw_floor,
                    ),
                    a.min_raw_cap,
                )
            )
            st.delta_on = max(
                a.sigma_delta_on * sd,
                a.delta_ratio * st.min_raw,             # 抬升量要与该 query 的量纲相称
                a.delta_on_floor,
            )

    def _maybe_reprobe(self, novelty: Optional[float], t: float) -> None:
        """场景突变 -> 重新探测。novelty 是免费的，本来就在算。"""
        a = self.cfg.negatives.auto
        if (
            self.cfg.negatives.mode != "auto"
            or not a.reprobe_on_scene_change
            or novelty is None
            or novelty < a.scene_change_novelty
        ):
            return
        self._neg_emb = None
        self._probe = []
        self._probe_t0 = t
        self._min_raw_override = None
        self._delta_on_override = None
        self.picked_negatives = []

    def _score_image(self, img: np.ndarray) -> np.ndarray:
        """对任意一张图，一次前向拿到全部 query 的分。裁剪块和全图共用这条路径。"""
        return self._scores(self.enc.encode_frame(img))

    def _roi_rescue(
        self, frame_rgb: np.ndarray, full: np.ndarray
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """全图分接近 min_raw 但没够到时，裁几块再看一眼。

        只提升「存在性」判断，不动 EMA/基线——那两个必须吃连续的全图分，
        否则裁与不裁之间会产生人为跳变，反倒制造误触发。
        """
        r, g = self.cfg.roi, self.cfg.gate
        _, _, min_raw = self.thresholds()
        band_lo = min_raw - r.margin
        if not np.any((full >= band_lo) & (full < min_raw)):
            return full, None

        h, w = frame_rgb.shape[:2]
        boxes = _grid_boxes(h, w, r.grid, r.frac)
        if r.max_crops:
            boxes = boxes[: r.max_crops]

        best = full.copy()
        best_box = None
        for (x, y, bw, bh) in boxes:
            s = self._score_image(frame_rgb[y : y + bh, x : x + bw])
            if s.max() > best.max():
                best_box = (x, y, bw, bh)
            best = np.maximum(best, s)
            if best.max() >= min_raw:        # 够了就停，别白扫
                break
        return best, best_box

    @property
    def cosine_mode(self) -> bool:
        """没有负样本就用纯余弦。零配置，换任何环境都能直接跑。"""
        return self._n_neg == 0

    def thresholds(self, state: Optional["_QueryState"] = None):
        """返回 (on, off, min_raw)。state 给定时用该 query 自己标定的值。"""
        g = self.cfg.gate
        if g.mode != "relative":
            return g.tau_on, g.tau_off, g.min_raw
        if self.cosine_mode:
            return g.cos_delta_on, g.cos_delta_on * 0.4, g.cos_min_raw
        if state is not None and state.delta_on is not None:
            return state.delta_on, state.delta_on * 0.4, state.min_raw
        return g.delta_on, g.delta_off, g.min_raw

    def clear_queries(self, keep_ids=None) -> int:
        """移除 query。keep_ids 里的保留。返回移除了几条。"""
        keep = set(keep_ids or [])
        kept = [(i, q) for i, q in enumerate(self.queries) if q.id in keep]
        removed = len(self.queries) - len(kept)
        if not removed:
            return 0
        idx = [i for i, _ in kept]
        self.queries = [q for _, q in kept]
        self._q_emb = self._q_emb[idx] if (idx and self._q_emb is not None) else None
        self._state = {q.id: self._state[q.id] for q in self.queries}
        return removed

    def _scores(self, frame_emb: torch.Tensor) -> np.ndarray:
        """每个 query 一个分数。

        有负样本：在 [该 query, *负样本] 上做 softmax，判别力更强但负样本跟环境绑定。
        没负样本：直接用余弦，判别力略弱但换环境不用配置。
        """
        sims = (self._text_emb @ frame_emb).cpu().numpy()
        n_q = len(self.queries)
        if self.cosine_mode:
            return sims[:n_q].astype(np.float32)
        negs = sims[n_q:] if self._n_neg else np.array([])
        scale = self.cfg.gate.softmax_scale
        out = np.empty(n_q, dtype=np.float32)
        for i in range(n_q):
            logits = np.concatenate(([sims[i]], negs)) * scale
            logits -= logits.max()
            e = np.exp(logits)
            out[i] = e[0] / e.sum()
        return out

    def step(self, frame_rgb: np.ndarray, t: float) -> List[WhenEvent]:
        """喂一帧，拿回这一帧产生的所有事件（每个 query 至多一条）。"""
        self._frame_i += 1
        # 注意：没有 query 时也要照常编码。
        # 一是让 auto 探测能先把环境攒起来，等你说出第一条 query 就能立刻用；
        # 二是提前 return 会让摄像头空转、日志里显示「no frames encoded」，很误导。
        emb = self.enc.encode_frame(frame_rgb)

        novelty = None
        if self._prev_frame_emb is not None:
            novelty = float(1.0 - torch.dot(emb, self._prev_frame_emb).item())
        self._prev_frame_emb = emb

        self._maybe_reprobe(novelty, t)

        if self.probing:
            if self._probe_t0 is None:
                self._probe_t0 = t
            self._probe.append(emb)
            if len(self._probe) > _MAX_PROBE_FRAMES:
                self._probe.pop(0)
            if t - self._probe_t0 >= self.cfg.negatives.auto.probe_seconds:
                self._finish_probe()      # 有没有 query 都先把环境定下来
            else:
                return []          # 探测期不出事件，此时还没有「正常」的定义

        if not self.queries or self._q_emb is None:
            return []

        raw = self._scores(emb)
        g = self.cfg.gate

        # presence 用（可能被 ROI 抬高的）分，EMA/基线始终用连续的全图分
        presence, roi_box = (
            self._roi_rescue(frame_rgb, raw) if self.cfg.roi.enabled else (raw, None)
        )
        events: List[WhenEvent] = []

        for i, q in enumerate(self.queries):
            st = self._state[q.id]
            st.update(float(raw[i]), g.ema_alpha, g.baseline_alpha)

            on_thr, off_thr, min_raw = self.thresholds(st)
            value = st.lift if g.mode == "relative" else st.fast

            fired = False
            warm = t >= g.warmup_s
            present = float(presence[i]) >= min_raw    # 「东西真的在画面里」
            if not st.on:
                if warm and present and value >= on_thr and t >= st.cooldown_until:
                    st.on = True
                    fired = True                       # 只在上升沿触发
                    st.cooldown_until = t + g.cooldown_s
                    # 事件已经报过了，这个新画面从此就是「正常」。
                    # 不把基线拉上来的话，慢 EMA 要爬 ~16 秒才追上，
                    # 这期间 lift 一直偏高，状态一抖就会反复重报同一件事。
                    if g.snap_baseline_on_fire:
                        st.slow = st.fast
            elif g.latch_until_absent:
                # 报过之后就锁住，直到目标真的从画面里消失才重新武装。
                # 只看 lift 的话，物体被手挡一下再露出来就会重报同一件事。
                if float(presence[i]) < min_raw * 0.9:
                    st.on = False
            elif value < off_thr:
                st.on = False

            if not fired and not g.emit_silent:
                continue

            window_start = max(0.0, t - 4.0 / max(g.fps, 1e-6))
            events.append(
                WhenEvent(
                    seq=self.next_seq(),
                    t_emit=round(t, 3),
                    route=Route.TRIGGER if fired else Route.SILENT,
                    trigger_type=q.trigger_type,
                    query=QueryRef(text=q.text, origin=_ORIGIN[q.trigger_type], query_id=q.id),
                    score=round(value, 4),
                    raw_score=round(float(raw[i]), 4),
                    threshold=on_thr,
                    baseline=round(st.slow, 4) if st.slow is not None else None,
                    evidence=Evidence(
                        window=[round(window_start, 3), round(t, 3)],
                        frame_idx=list(range(max(0, self._frame_i - 4), self._frame_i + 1)),
                        novelty=round(novelty, 4) if novelty is not None else None,
                        roi_box=list(roi_box) if roi_box else None,
                    ),
                    urgency=q.urgency,
                    cooldown_until=round(st.cooldown_until, 3) if fired else None,
                )
            )
        return events


def emit_instant(seq: int, t: float, text: str, audio_rms: Optional[float] = None) -> WhenEvent:
    """INSTANT 通路：用户开口提问，WHEN 不做任何视觉判断，直接放行给 WHICH。"""
    return WhenEvent(
        seq=seq,
        t_emit=round(t, 3),
        route=Route.TRIGGER,
        trigger_type=TriggerType.INSTANT,
        query=QueryRef(text=text, origin=QueryOrigin.USER_SPEECH, query_id=None),
        score=1.0,
        raw_score=1.0,
        threshold=0.0,
        evidence=Evidence(window=[round(t, 3), round(t, 3)], audio_rms=audio_rms),
        urgency=Urgency.NORMAL,
    )


__all__ = ["VisualGate", "emit_instant"]
