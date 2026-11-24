# Copyright 2022 Tsinghua University
# SPDX-License-Identifier: Apache-2.0
# Adapted for inference from Whistle (CAT) and CTC-TTS. See NOTICE.

import math
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from threading import Lock

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..utils.config import read_encoder_config


CHECKPOINT_LOAD_LOCK = Lock()


class VGG2LSubsampling(nn.Module):
    def __init__(self, in_channel: int = 1):
        super().__init__()
        self.in_channel = in_channel
        self.vgg_conv = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),
        )

    def forward(self, x: Tensor, lens: Tensor) -> tuple[Tensor, Tensor]:
        batch, frames, features = x.shape
        x = x.reshape(batch, frames, self.in_channel, features // self.in_channel)
        x = self.vgg_conv(x.transpose(1, 2)).transpose(1, 2).flatten(2)
        return x, (lens + 3) // 4


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, reverse: bool = False):
        super().__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.register_buffer("pe", self.extend_pe(max_len), persistent=False)

    def extend_pe(self, length: int) -> Tensor:
        position = torch.arange(length, dtype=torch.float32)
        if self.reverse:
            position = position.flip(0)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        angles = position[:, None] * div_term
        return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)

    def forward(self, x: Tensor) -> Tensor:
        frames = x.size(1)
        if frames > self.pe.size(0):
            return self.extend_pe(frames).to(x)
        return self.pe[:frames]


class LegacyRelPositionalEncoding(PositionalEncoding):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__(d_model, max_len=max_len, reverse=True)


def rel_shift(scores: Tensor) -> Tensor:
    heads_shape = scores.shape[:-2]
    queries, keys = scores.shape[-2:]
    padded = F.pad(scores, (1, 0)).reshape(*heads_shape, keys + 1, queries)
    return padded[..., 1:, :].reshape_as(scores)


class RelPositionMultiHeadAttention(nn.Module):
    _rel_shift = staticmethod(rel_shift)

    def __init__(self, idim: int, n_head: int, d_head: int):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_head
        self.scale = d_head ** -0.5
        self.call_qkv = nn.Linear(idim, 3 * n_head * d_head, bias=False)
        self.linearpos = nn.Linear(idim, n_head * d_head, bias=False)
        self.linearout = nn.Linear(n_head * d_head, idim, bias=False)
        self.r_r_bias = nn.Parameter(torch.empty(n_head, d_head))
        self.r_w_bias = nn.Parameter(torch.empty(n_head, d_head))
        nn.init.xavier_uniform_(self.r_r_bias)
        nn.init.xavier_uniform_(self.r_w_bias)

    def forward(self, word_embedding: Tensor, pos_emb: Tensor, lens: Tensor) -> Tensor:
        batch, frames, _ = word_embedding.shape
        query, key, value = (
            item.reshape(batch, frames, self.n_head, self.d_head).transpose(1, 2)
            for item in self.call_qkv(word_embedding).chunk(3, dim=-1)
        )
        position = self.linearpos(pos_emb).reshape(frames, self.n_head, self.d_head).transpose(0, 1)
        content = (query + self.r_w_bias[:, None]) @ key.transpose(-2, -1)
        relative = (query + self.r_r_bias[:, None]) @ position.transpose(-2, -1)
        attn_score = (content + self._rel_shift(relative)) * self.scale
        attn_mask = torch.arange(frames, device=word_embedding.device)[None, :] >= lens[:, None]
        attn_score = attn_score.masked_fill(attn_mask[:, None, None, :], float("-inf"))
        attn_vec = attn_score.softmax(dim=-1) @ value
        return self.linearout(attn_vec.transpose(1, 2).flatten(2))


class StandardRelPositionalMultiHeadAttention(RelPositionMultiHeadAttention):
    def __init__(self, idim: int, n_head: int):
        super().__init__(idim, n_head, idim // n_head)


class LegacyRelPositionMultiHeadedAttention(nn.Module):
    rel_shift = staticmethod(rel_shift)

    def __init__(self, n_head: int, n_feat: int):
        super().__init__()
        self.h = n_head
        self.d_k = n_feat // n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.empty(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.empty(self.h, self.d_k))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

    def forward(self, x: Tensor, pos_emb: Tensor) -> Tensor:
        batch, frames, _ = x.shape
        q, k, v = (
            projection(x).reshape(batch, frames, self.h, self.d_k).transpose(1, 2)
            for projection in (self.linear_q, self.linear_k, self.linear_v)
        )
        p = self.linear_pos(pos_emb).reshape(frames, self.h, self.d_k).transpose(0, 1)
        matrix_ac = (q + self.pos_bias_u[:, None]) @ k.transpose(-2, -1)
        matrix_bd = (q + self.pos_bias_v[:, None]) @ p.transpose(-2, -1)
        scores = (matrix_ac + self.rel_shift(matrix_bd)) / self.d_k ** 0.5
        x = scores.softmax(dim=-1) @ v
        return self.linear_out(x.transpose(1, 2).flatten(2))


class FFModule(nn.Module):
    def __init__(self, idim: int, res_factor: float):
        super().__init__()
        self._res_factor = res_factor
        self.ln = nn.LayerNorm(idim)
        self.fc0 = nn.Linear(idim, idim * 4)
        self.fc1 = nn.Linear(idim * 4, idim)

    def forward(self, x: Tensor) -> Tensor:
        hidden = F.silu(self.fc0(self.ln(x)))
        return x + self._res_factor * self.fc1(hidden)


class ConvModule(nn.Module):
    def __init__(self, idim: int, kernel_size: int, multiplier: int, causal: bool):
        super().__init__()
        self.ln = nn.LayerNorm(idim)
        self.pointwise_conv0 = nn.Conv1d(idim, idim * 2, 1)
        left = 0 if causal else (kernel_size - 1) // 2
        self.padding = (left, kernel_size - 1 - left)
        self.depthwise_conv = nn.Conv1d(idim, idim * multiplier, kernel_size, groups=idim)
        self.bn = nn.BatchNorm1d(idim * multiplier)
        self.pointwise_conv1 = nn.Conv1d(idim * multiplier, idim, 1)

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.pointwise_conv0(self.ln(x).transpose(1, 2))
        hidden = self.depthwise_conv(F.pad(F.glu(hidden, dim=1), self.padding))
        if hidden.shape[0] > 1 or hidden.shape[2] > 1:
            hidden = self.bn(hidden)
        hidden = self.pointwise_conv1(F.silu(hidden))
        return x + hidden.transpose(1, 2)


class MHSAModule(nn.Module):
    def __init__(self, hdim: int, d_head: int, num_heads: int):
        super().__init__()
        self.ln = nn.LayerNorm(hdim)
        if d_head == -1:
            self.mha = StandardRelPositionalMultiHeadAttention(hdim, num_heads)
        else:
            self.mha = RelPositionMultiHeadAttention(hdim, num_heads, d_head)

    def forward(self, x: Tensor, pos_emb: Tensor, lens: Tensor) -> Tensor:
        return x + self.mha(self.ln(x), pos_emb, lens)


class ConformerCell(nn.Module):
    def __init__(
        self, idim: int, res_factor: float, d_head: int, num_heads: int, kernel_size: int,
        multiplier: int, causal: bool, modernized_atten: bool,
    ):
        super().__init__()
        self.modernized_atten = modernized_atten
        self.ffm0 = FFModule(idim, res_factor)
        if modernized_atten:
            self.mhsam = LegacyRelPositionMultiHeadedAttention(num_heads, idim)
        else:
            self.mhsam = MHSAModule(idim, d_head, num_heads)
        self.convm = ConvModule(idim, kernel_size, multiplier, causal)
        self.ffm1 = FFModule(idim, res_factor)
        self.ln = nn.LayerNorm(idim)

    def forward(self, x: Tensor, pos_emb: Tensor, lens: Tensor) -> Tensor:
        x = self.ffm0(x)
        if self.modernized_atten:
            x = x + self.mhsam(self.ln(x), pos_emb)
        else:
            x = self.mhsam(x, pos_emb, lens)
        return self.ln(self.ffm1(self.convm(x)))


class ConformerNet(nn.Module):
    def __init__(
        self, num_cells: int, idim: int, hdim: int, num_classes: int,
        conv: str = "vgg2l", res_factor: float = 0.5,
        d_head: int = -1, num_heads: int = 4, kernel_size: int = 32,
        multiplier: int = 1,
        delta_feats: bool = False, causal: bool = False, modernized_atten: bool = False,
        with_head: bool = True, frame_rate: float = 25.0,
    ):
        super().__init__()
        if conv != "vgg2l" or not with_head:
            raise ValueError("This aligner requires a VGG2L encoder with a CTC classifier")
        if num_heads < 1 or hdim < 1 or hdim % 2 or hdim % num_heads:
            raise ValueError("The encoder hidden size must be even and divisible by the number of heads")
        if not math.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError("The encoder frame rate must be finite and positive")
        self.idim = idim
        self.delta_feats = delta_feats
        self.frame_rate = frame_rate
        in_channel = 3 if delta_feats else 1
        self.conv_subsampling = VGG2LSubsampling(in_channel)
        self.classifier = nn.Linear(hdim, num_classes)
        self.linear_drop = nn.Sequential(OrderedDict(
            linear=nn.Linear(128 * ((idim // in_channel + 3) // 4), hdim),
        ))
        self.cells = nn.ModuleList(
            ConformerCell(
                hdim, res_factor, d_head, num_heads, kernel_size, multiplier,
                causal, modernized_atten,
            )
            for _ in range(num_cells)
        )
        self.pe = LegacyRelPositionalEncoding(hdim) if modernized_atten else PositionalEncoding(hdim)

    def impl_forward(self, x: Tensor, lens: Tensor) -> tuple[Tensor, Tensor]:
        x, lens = self.conv_subsampling(x, lens)
        x = self.linear_drop(x)
        pos_emb = self.pe(x)
        for cell in self.cells:
            x = cell(x, pos_emb, lens)
        return x, lens

    def forward(self, x: Tensor, lens: Tensor) -> tuple[Tensor, Tensor]:
        x, lens = self.impl_forward(x, lens)
        return self.classifier(x), lens


def load_pretrained_encoder(
    checkpoint: str | Path,
    device: str | torch.device,
    num_classes: int,
    config: str | Path | Mapping | None = None,
) -> ConformerNet:
    """Load a Conformer from a local checkpoint and encoder configuration."""
    encoder = read_encoder_config(config)
    if encoder.get("type") != "ConformerNet":
        raise ValueError("This aligner requires a ConformerNet encoder configuration")
    kwargs = dict(encoder["kwargs"])
    if kwargs.setdefault("num_classes", num_classes) != num_classes:
        raise ValueError("The tokenizer vocabulary size must match the encoder configuration")
    for key in ("dropout", "dropout_in", "dropout_attn"):
        kwargs.pop(key, None)
    model = ConformerNet(frame_rate=float(encoder.get("tokens_per_second", 25.0)), **kwargs)

    # Older CAT checkpoints serialized the typing alias for OrderedDict.
    with CHECKPOINT_LOAD_LOCK, torch.serialization.safe_globals([(OrderedDict, "typing.OrderedDict")]):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError("The checkpoint must be a state dictionary")
    state = state.get("model", state)
    if not isinstance(state, Mapping):
        raise ValueError("The checkpoint's 'model' entry must be a state dictionary")
    state = {
        key.removeprefix("module.").replace(".mhsencoder.", ".mhsam."): value
        for key, value in state.items()
    }
    for prefix in ("s2p_model.", "encoder.", "am."):
        if any(key.startswith(prefix) for key in state):
            state = {
                key.removeprefix(prefix): value
                for key, value in state.items()
                if key.startswith(prefix)
            }
            break
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
