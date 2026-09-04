"""Smoke tests for the custom multi-head attention layer."""

import torch

from pkg.train.model.transformer import MultiHeadAttention


def test_basic_functionality():
    """Verify configuration properties and output tensor shape."""
    config = {"d_model": 512, "n_heads": 8, "dropout": 0.1}

    attention = MultiHeadAttention(config)
    print(f"  - d_model: {attention.d_model}")
    print(f"  - n_heads: {attention.n_heads}")
    print(f"  - d_k: {attention.d_k}")

    batch_size = 2
    seq_len = 10
    d_model = 512

    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)

    print(f"  - query shape: {query.shape}")
    print(f"  - key shape: {key.shape}")
    print(f"  - value shape: {value.shape}")

    output = attention(query, key, value)

    print(f"  - output shape: {output.shape}")

    assert output.shape == (batch_size, seq_len, d_model), f"wrong shape: {output.shape}"


if __name__ == "__main__":
    test_basic_functionality()
