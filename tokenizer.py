from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import mlx.core as mx
from jinja2 import Environment
from tokenizers import Tokenizer

from .defaults import TOKENIZER_FALLBACK_DIR


@dataclass
class Qwen3Tokenizer:
    tokenizer: Tokenizer
    chat_template: str
    pad_id: int
    eos_id: int
    _compiled_template: object = None

    def __post_init__(self):
        env = Environment()
        self._compiled_template = env.from_string(self.chat_template)

    @classmethod
    def from_repo(cls, repo_path: Path) -> "Qwen3Tokenizer":
        tok_dir = repo_path / "tokenizer"
        tok_path = tok_dir / "tokenizer.json"
        template_path = tok_dir / "chat_template.jinja"
        if not tok_path.exists():
            base_dir = Path.cwd() / TOKENIZER_FALLBACK_DIR / "tokenizer"
            if base_dir.exists():
                tok_dir = base_dir
                tok_path = tok_dir / "tokenizer.json"
                template_path = tok_dir / "chat_template.jinja"
        if not tok_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {tok_path}")
        tokenizer = Tokenizer.from_file(str(tok_path))
        chat_template = template_path.read_text()

        pad_token = "<|endoftext|>"
        eos_token = "<|im_end|>"
        special_map = tok_dir / "special_tokens_map.json"
        if special_map.exists():
            data = json.loads(special_map.read_text())
            pad_token = data.get("pad_token", {}).get("content", pad_token)
            eos_token = data.get("eos_token", {}).get("content", eos_token)
        pad_id = tokenizer.token_to_id(pad_token)
        eos_id = tokenizer.token_to_id(eos_token)
        if pad_id is None or eos_id is None:
            raise ValueError("Tokenizer missing required special tokens")
        return cls(tokenizer=tokenizer, chat_template=chat_template, pad_id=pad_id, eos_id=eos_id)

    def apply_chat_template(self, prompt: str, add_generation_prompt: bool = True) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self._compiled_template.render(
            messages=messages,
            tools=None,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )

    def encode_batch(
        self, prompts: Iterable[str], max_length: int, pad_to_max: bool = False
    ) -> Tuple[mx.array, mx.array]:
        input_ids: List[List[int]] = []
        attention_masks: List[List[int]] = []
        for prompt in prompts:
            text = self.apply_chat_template(prompt, add_generation_prompt=True)
            enc = self.tokenizer.encode(text)
            ids = enc.ids
            if len(ids) > max_length:
                ids = ids[:max_length]
            mask = [1] * len(ids)
            input_ids.append(ids)
            attention_masks.append(mask)

        actual_max = max(len(ids) for ids in input_ids)
        if pad_to_max:
            target_len = max_length
        else:
            target_len = min(max_length, ((actual_max + 63) // 64) * 64)

        for i in range(len(input_ids)):
            pad_len = target_len - len(input_ids[i])
            if pad_len > 0:
                input_ids[i] = input_ids[i] + [self.pad_id] * pad_len
                attention_masks[i] = attention_masks[i] + [0] * pad_len

        return mx.array(input_ids, dtype=mx.int32), mx.array(attention_masks, dtype=mx.int32)
