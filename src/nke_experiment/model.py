"""Variant A: trainable full-snapshot decision model. Requires the model extra.

The released model must include trained head weights. The pretrained encoder alone
is not an NKE checkpoint. This module does not implement the recurrent C variant.
"""

from __future__ import annotations

import json
from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


def record_text(entity_id: str, record: dict[str, Any]) -> str:
    return "entity=" + entity_id + " " + json.dumps(record, sort_keys=True, separators=(",", ":"))


class SnapshotDecisionModel(nn.Module):
    def __init__(self, backbone: str = "answerdotai/ModernBERT-base", width: int = 256, freeze_backbone: bool = True):
        super().__init__()
        self.backbone_id = backbone
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.encoder = AutoModel.from_pretrained(backbone)
        if freeze_backbone:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        hidden = self.encoder.config.hidden_size
        self.project = nn.Linear(hidden, width)
        self.record_queries = nn.Parameter(torch.randn(4, width) * 0.02)
        self.record_pool = nn.MultiheadAttention(width, 4, batch_first=True)
        self.cross = nn.ModuleList([nn.MultiheadAttention(width, 4, batch_first=True) for _ in range(2)])
        self.norm = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.scorer = nn.Linear(width, 1)

    def _encode(self, texts: list[str], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        if not any(p.requires_grad for p in self.encoder.parameters()):
            with torch.no_grad():
                hidden = self.encoder(**batch).last_hidden_state
        else:
            hidden = self.encoder(**batch).last_hidden_state
        return self.project(hidden), batch["attention_mask"].bool()

    def forward(self, row: dict[str, Any]) -> torch.Tensor:
        records = row["state"]["records"]
        if not 1 <= len(records) <= 64:
            raise ValueError("supports 1–64 state records")
        candidates = list(row["candidates"]) + ["none"]
        if not 2 <= len(candidates) <= 33:
            raise ValueError("supports 1–32 supplied candidates")
        record_strings = [record_text(k, v) for k, v in sorted(records.items())]
        record_tokens, record_mask = self._encode(record_strings, max_length=128)
        query = self.record_queries.unsqueeze(0).expand(len(records), -1, -1)
        memory, _ = self.record_pool(query, record_tokens, record_tokens, key_padding_mask=~record_mask,
                                      need_weights=False)
        memory = memory.reshape(1, -1, memory.shape[-1])
        candidate_texts = [row["question"] + " Candidate: " + c for c in candidates]
        candidate_tokens, candidate_mask = self._encode(candidate_texts, max_length=128)
        denom = candidate_mask.sum(dim=1, keepdim=True).clamp(min=1)
        candidate_vectors = (candidate_tokens * candidate_mask.unsqueeze(-1)).sum(dim=1) / denom
        candidate_vectors = candidate_vectors.unsqueeze(0)
        for cross, norm in zip(self.cross, self.norm):
            update, _ = cross(candidate_vectors, memory, memory, need_weights=False)
            candidate_vectors = norm(candidate_vectors + update)
        logits = self.scorer(candidate_vectors).squeeze(0).squeeze(-1)
        eligible = row.get("eligible")
        if eligible is not None:
            if set(eligible) - set(row["candidates"]):
                raise ValueError("eligible mask contains unknown candidate")
            allowed = torch.tensor([c in eligible for c in candidates], device=logits.device)
            allowed[-1] = True  # 'none' remains available when all actions are masked.
            logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
        return logits

    @torch.no_grad()
    def decide(self, row: dict[str, Any]) -> dict[str, Any]:
        was_training = self.training
        self.eval()
        try:
            logits = self(row)
            probabilities = torch.softmax(logits.float(), dim=-1).cpu().tolist()
        finally:
            self.train(was_training)
        candidates = list(row["candidates"]) + ["none"]
        return {"choice": candidates[max(range(len(probabilities)), key=probabilities.__getitem__)],
                "probabilities": dict(zip(candidates, probabilities)), "state_version": row["state"]["version"]}
