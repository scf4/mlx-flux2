"""Tests for tokenizer functionality."""

import pytest

# Note: Full tokenizer tests require a tokenizer.json file.
# These tests verify the interface and error handling.


def test_encode_batch_empty_raises():
    """Test that encode_batch raises ValueError for empty prompts list."""
    from unittest.mock import MagicMock, patch

    from flux2_mlx.tokenizer import Qwen3Tokenizer

    with patch.object(Qwen3Tokenizer, "__init__", lambda self: None):
        tokenizer = Qwen3Tokenizer()
        tokenizer.tokenizer = MagicMock()
        tokenizer.pad_id = 0
        tokenizer._compiled_template = MagicMock()

        with pytest.raises(ValueError, match="requires at least one prompt"):
            tokenizer.encode_batch([], max_length=512)


def test_tokenizer_dataclass_fields():
    """Test that Qwen3Tokenizer has expected fields."""
    from flux2_mlx.tokenizer import Qwen3Tokenizer
    import dataclasses

    fields = {f.name for f in dataclasses.fields(Qwen3Tokenizer)}
    assert "tokenizer" in fields
    assert "chat_template" in fields
    assert "pad_id" in fields
    assert "eos_id" in fields
