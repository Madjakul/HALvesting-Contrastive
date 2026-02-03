# halvesting_contrastive/modules/modeling_halvesting_contrastive.py

import logging
from typing import TYPE_CHECKING, Any, Dict

import lightning as L
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from transformers import get_cosine_schedule_with_warmup

from halvesting_contrastive.modules.info_nce_loss import InfoNCELoss
from halvesting_contrastive.modules.language_model import LanguageModel
from halvesting_contrastive.utils.helpers import flatten_dict

if TYPE_CHECKING:
    from halvesting_contrastive.utils.configs import BaseConfig


class Retriever(L.LightningModule):

    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__()
        flat_params = flatten_dict(cfg.to_dict())
        self.save_hyperparameters(flat_params)
        self.cfg = cfg
        self.lm = LanguageModel(cfg)
        self.contrastive_loss = InfoNCELoss(cfg)

    def configure_optimizers(self) -> Dict[str, Any]:
        logging.info(
            f"""Configuring optimizer: AdamW with lr={self.cfg.train.lr},
             weight_decay={self.cfg.train.weight_decay}."""
        )
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = max(1, int(0.1 * total_steps))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5,
            last_epoch=-1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch seq"],
        attention_mask: Int[torch.Tensor, "batch seq"],
        **kwargs: Any,
    ) -> Float[torch.Tensor, "batch seq hidden"]:
        return self.lm(input_ids=input_ids, attention_mask=attention_mask)

    # --- ADDED HELPER ---
    def _pool_embeddings(
        self,
        embs: Float[torch.Tensor, "batch seq hidden"],
        mask: Int[torch.Tensor, "batch seq"],
    ) -> Float[torch.Tensor, "batch hidden"]:
        vec = (embs * mask.unsqueeze(-1)).sum(dim=1)
        vec = F.normalize(vec, p=2, dim=-1)
        return vec

    def training_step(self, batch, batch_idx: int) -> Float[torch.Tensor, ""]:
        q_embs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        pos_embs = self(
            input_ids=batch["pos_input_ids"],
            attention_mask=batch["pos_attention_mask"],
        )
        neg_embs = self(
            input_ids=batch["neg_input_ids"],
            attention_mask=batch["neg_attention_mask"],
        )
        q_mask = batch["attention_mask"]
        batch_size = q_embs.size(0)

        if self.trainer.world_size > 1 and self.cfg.train.gather:
            # all_gather adds a dimension at the start, so we flatten it with the batch dim
            # Shape changes from [num_gpus, batch_size, seq, hidden] -> [global_batch_size, seq, hidden]
            targets = (
                torch.arange(batch_size, device=q_embs.device)
                + batch_size * self.trainer.global_rank
            )
            all_pos_embs = self.all_gather(pos_embs, sync_grads=True).flatten(0, 1)
            all_pos_mask = self.all_gather(batch["pos_attention_mask"]).flatten(0, 1)
            all_neg_embs = self.all_gather(neg_embs, sync_grads=True).flatten(0, 1)
            all_neg_mask = self.all_gather(batch["neg_attention_mask"]).flatten(0, 1)

            k_embs = torch.cat([all_pos_embs, all_neg_embs], dim=0)
            k_mask = torch.cat([all_pos_mask, all_neg_mask], dim=0)
            loss_metrics = self.contrastive_loss(
                query_embs=q_embs,
                key_embs=k_embs,
                q_mask=q_mask,
                k_mask=k_mask,
                targets=targets,
            )
        else:
            targets = torch.arange(batch_size, device=q_embs.device)
            k_embs = torch.cat([pos_embs, neg_embs], dim=0)
            k_mask = torch.cat(
                [batch["pos_attention_mask"], batch["neg_attention_mask"]], dim=0
            )
            loss_metrics = self.contrastive_loss(
                query_embs=q_embs,
                key_embs=k_embs,
                q_mask=q_mask,
                k_mask=k_mask,
                targets=targets,
            )

        loss = loss_metrics["loss"]

        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.cfg.data.batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        q_embs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        pos_embs = self(
            input_ids=batch["pos_input_ids"],
            attention_mask=batch["pos_attention_mask"],
        )
        neg_embs = self(
            input_ids=batch["neg_input_ids"],
            attention_mask=batch["neg_attention_mask"],
        )

        q_vec = self._pool_embeddings(q_embs, batch["attention_mask"])
        pos_vec = self._pool_embeddings(pos_embs, batch["pos_attention_mask"])
        neg_vec = self._pool_embeddings(neg_embs, batch["neg_attention_mask"])

        return {
            "q_vec": q_vec,
            "pos_vec": pos_vec,
            "neg_vec": neg_vec,
            "index": batch["index"],
        }

    def test_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        q_embs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        pos_embs = self(
            input_ids=batch["pos_input_ids"],
            attention_mask=batch["pos_attention_mask"],
        )
        neg_embs = self(
            input_ids=batch["neg_input_ids"],
            attention_mask=batch["neg_attention_mask"],
        )

        q_vec = self._pool_embeddings(q_embs, batch["attention_mask"])
        pos_vec = self._pool_embeddings(pos_embs, batch["pos_attention_mask"])
        neg_vec = self._pool_embeddings(neg_embs, batch["neg_attention_mask"])

        return {
            "q_vec": q_vec,
            "pos_vec": pos_vec,
            "neg_vec": neg_vec,
        }

