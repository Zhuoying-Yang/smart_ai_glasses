"""SigLIP 编码器：把帧和 query 文本映射到同一个空间。

WHEN 层之所以能零训练就做到 query-conditioned，全靠 SigLIP 自带文本塔
——帧和文字可以直接算相似度。
"""

from __future__ import annotations

import time
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from .config import ModelCfg


def _as_tensor(out) -> torch.Tensor:
    """transformers 5 的 get_*_features 返回 output 对象，4.x 返回张量。两种都吃。"""
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("pooler_output", "last_hidden_state", "image_embeds", "text_embeds"):
        val = getattr(out, attr, None)
        if isinstance(val, torch.Tensor):
            return val
    raise TypeError(f"无法从 {type(out).__name__} 取出特征张量")


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SiglipEncoder:
    def __init__(self, cfg: ModelCfg):
        self.device = _resolve_device(cfg.device)
        # MPS 上 fp16 偶有算子回退问题；CPU 一律用 fp32。
        self.dtype = torch.float32 if self.device == "cpu" else getattr(torch, cfg.dtype)

        t0 = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(cfg.siglip)
        self.model = (
            AutoModel.from_pretrained(cfg.siglip, dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        self.load_s = time.perf_counter() - t0

        self.frame_times: List[float] = []

    @property
    def logit_scale(self) -> float:
        return float(self.model.logit_scale.exp().item())

    @property
    def logit_bias(self) -> float:
        return float(self.model.logit_bias.item())

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        """返回 L2 归一化的文本特征 [N, D]。query 是静态的，只需编码一次。"""
        batch = self.processor(
            text=list(texts),
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(self.device)
        feats = _as_tensor(self.model.get_text_features(**batch))
        return torch.nn.functional.normalize(feats.float(), dim=-1)

    @torch.no_grad()
    def encode_frame(self, frame_rgb: np.ndarray) -> torch.Tensor:
        """frame_rgb: HxWx3 uint8 RGB。返回 L2 归一化的图像特征 [D]。"""
        t0 = time.perf_counter()
        batch = self.processor(
            images=Image.fromarray(frame_rgb), return_tensors="pt"
        ).to(self.device, self.dtype)
        feats = _as_tensor(self.model.get_image_features(**batch))
        out = torch.nn.functional.normalize(feats.float(), dim=-1)[0]
        if self.device == "mps":
            torch.mps.synchronize()
        self.frame_times.append(time.perf_counter() - t0)
        return out

    def timing_report(self) -> str:
        if not self.frame_times:
            return "no frames encoded"
        arr = np.asarray(self.frame_times)
        return (
            f"frames={arr.size}  mean={arr.mean()*1000:.1f}ms  "
            f"p50={np.percentile(arr,50)*1000:.1f}ms  "
            f"p95={np.percentile(arr,95)*1000:.1f}ms  "
            f"max_fps={1.0/arr.mean():.2f}"
        )


__all__ = ["SiglipEncoder"]
