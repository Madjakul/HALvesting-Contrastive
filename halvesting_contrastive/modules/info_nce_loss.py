# halvesting_contrastive/modules/base_loss.py

import logging
from typing import TYPE_CHECKING, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int

if TYPE_CHECKING:
    from halvesting_contrastive.utils.configs import BaseConfig


class InfoNCELoss(nn.Module):

    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__()
        self.cfg = cfg
        logging.info(f"Using InfoNCE Loss with tau={self.cfg.train.tau}")
        self.register_buffer("tau", torch.tensor(self.cfg.train.tau))

    def forward(
        self,
        query_embs: Float[torch.Tensor, "batch seq hidden"],
        key_embs: Float[torch.Tensor, "n_times_batch seq hidden"],
        q_mask: Int[torch.Tensor, "batch seq"],
        k_mask: Int[torch.Tensor, "n_times_batch seq"],
        targets: Int[torch.Tensor, "n_times_batch"],
    ) -> Dict[str, torch.Tensor]:
        all_scores = self.sum_pooling(
            query_embs=query_embs,
            key_embs=key_embs,
            q_mask=q_mask,
            k_mask=k_mask,
        )
        all_scaled_scores = all_scores / self.tau  # type: ignore
        local_targets = torch.arange(query_embs.size(0), device=query_embs.device)

        poss = all_scores[local_targets, targets]
        negs = all_scores[local_targets, targets + key_embs.size(0) // 2]

        loss = F.cross_entropy(all_scaled_scores, targets, reduction="mean")

        return {
            "all_scores": all_scores,
            "poss": poss,
            "negs": negs,
            "loss": loss,
        }

    @staticmethod
    def sum_pooling(
        query_embs: Float[torch.Tensor, "batch seq hidden"],
        key_embs: Float[torch.Tensor, "two_times_batch seq hidden"],
        q_mask: Int[torch.Tensor, "batch seq"],
        k_mask: Int[torch.Tensor, "two_times_batch seq"],
    ) -> Float[torch.Tensor, "batch two_times_batch"]:
        # Mean pooling and normalization
        query_vec = (query_embs * q_mask.unsqueeze(-1)).sum(dim=1)
        query_vec = F.normalize(query_vec, p=2, dim=-1)

        key_vec = (key_embs * k_mask.unsqueeze(-1)).sum(dim=1)
        key_vec = F.normalize(key_vec, p=2, dim=-1)

        all_scores = torch.matmul(query_vec, key_vec.T)
        return all_scores
