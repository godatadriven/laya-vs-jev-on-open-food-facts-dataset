"""Laya's truncation, counted on our side: the strict server (`LAYA_JEV_STRICT=1`) no longer reports it.

Laya puts each question's head (its instructions and one span per option) in front of the state and
keeps only the state tokens that still fit in `max_len`. This repeats that with Laya's own code and
the checkpoint's own tokenizer, so the counts are the ones the server would have reported.
"""

import importlib.metadata
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from laya.agent import Agent, _fix_tokenizer_config, _load_tokenizer
from laya.common import serialize_state, state_room

REPO = "convaiinnovations/laya"  # the `english` checkpoint the server loads


class Truncation:
    def __init__(self, revision: str, version: str, checkpoint: Path | None = None):
        """`revision` is the checkpoint commit from `/health`; `version` the Laya release the server runs.

        A local `checkpoint`, such as the fine-tuned one, is read from its folder: it carries its own
        tokenizer and token budgets.
        """
        installed = importlib.metadata.version("laya")
        if installed != version:
            raise RuntimeError(f"laya {installed} is installed, but the server runs {version}")
        if checkpoint:
            model_dir = str(checkpoint)
        else:
            model_dir = snapshot_download(
                REPO,
                revision=revision,
                allow_patterns=["rl_agent_config.json", "tokenizer/*"],
                local_files_only=True,
            )
        _fix_tokenizer_config(model_dir)  # what Laya does to the checkpoint before loading it
        with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
            cfg = json.load(f)
        self.tok = _load_tokenizer(os.path.join(model_dir, "tokenizer"), cfg)
        self.max_len = cfg.get("max_len", 512)
        self.head_max_len = cfg.get("head_max_len", 192)

    def room(self, question: dict) -> int:
        """How many state tokens `question` leaves inside `max_len`."""
        return state_room(self.tok, Agent._to_internal(question), self.max_len, self.head_max_len)

    def state_tokens(self, states: list[str]) -> list[int]:
        """The length of every state in tokens, tokenized the way the server does it."""
        texts = [serialize_state(state).replace(self.tok.mask_token, " ") for state in states]
        return [len(ids) for ids in self.tok(texts, add_special_tokens=False)["input_ids"]]
