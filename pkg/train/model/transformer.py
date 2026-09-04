"""Transformer attention components used by graph-model experiments."""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pkg.train.model.base_model import BaseModule


class MultiHeadAttention(BaseModule):
    """Scaled dot-product attention split across multiple representation heads."""

    def __init__(self, config: Dict, *args, **kwargs) -> None:
        """Validate dimensions and initialize attention projections."""
        super().__init__(config, *args, **kwargs)
        self._d_model: int = config["d_model"]
        self._n_heads: int = config["n_heads"]
        self._dropout_rate: float = config["dropout"]
        self._d_k: int = self._d_model // self._n_heads

        assert self._d_model % self._n_heads == 0

        self._init_graph()

    @property
    def d_model(self) -> int:
        """Return the full embedding width."""
        return self._d_model

    @property
    def n_heads(self) -> int:
        """Return the number of parallel attention heads."""
        return self._n_heads

    @property
    def _drop_rate(self) -> float:
        """Return the dropout probability applied to attention weights."""
        return self._dropout_rate

    @property
    def d_k(self) -> int:
        """Return the embedding width assigned to each attention head."""
        return self._d_k

    def _init_graph(self) -> None:
        """Create query, key, value, output, and dropout layers."""
        self._w_q: nn.Module = nn.Linear(self._d_model, self._d_model)
        self._w_k: nn.Module = nn.Linear(self._d_model, self._d_model)
        self._w_v: nn.Module = nn.Linear(self._d_model, self._d_model)
        self._w_o: nn.Module = nn.Linear(self._d_model, self._d_model)

        self._dropout: nn.Module = nn.Dropout(self._dropout_rate)
        self._scale: float = math.sqrt(self._d_k)

    def get_config(self) -> Dict:
        """Return the resolved attention configuration."""
        config = super().get_config()

        module_config = {
            "d_model": self._d_model,
            "n_heads": self._n_heads,
            "dropout_rate": self._dropout_rate,
            "d_k": self._d_k,
        }

        config.update(module_config)

        return config

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask=None) -> Tensor:
        """Apply multi-head scaled dot-product attention."""
        batch_size, seq, _ = query.shape  # batch_size, seq, _

        # (batch_size, seq, emb) => (batch_size, seq, n_head, sub_emb) => (batch_size, n_head, seq, sub_emb)
        q = self._w_q(query).view(batch_size, seq, self._n_heads, self._d_k).transpose(1, 2)
        k = self._w_k(key).view(batch_size, -1, self._n_heads, self._d_k).transpose(1, 2)
        v = self._w_v(value).view(batch_size, -1, self._n_heads, self._d_k).transpose(1, 2)

        # K: (batch_size, n_head, seq, sub_emb) => (batch_size, n_head, sub_emb, seq)
        # Q * K =>  (batch_size, n_head, seq, seq)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self._scale

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, n_head, seq, seq)
        attention_weights = self._dropout(attention_weights)

        context = torch.matmul(attention_weights, v)  # (batch_size, n_head, seq, sub_emb)

        # (batch_size, n_head, seq, sub_emb) => (batch_size, seq, n_head, sub_emb)
        # => (batch_size, seq, emb)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq, self._d_model)

        output = self._w_o(context)  # (batch_size, seq, emb)

        return output
