# halvesting_contrastive/modules/modeling_halvesting_contrastive.py

import logging
from typing import TYPE_CHECKING, Any, Dict, List

import lightning as L
import torch
import torch.nn.functional as F  # <-- ADDED
from jaxtyping import Float, Int
from torcheval.metrics import ReciprocalRank
from torchmetrics.retrieval import RetrievalNormalizedDCG, RetrievalRecall
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from halvesting_contrastive.modules.info_nce_loss import InfoNCELoss
from halvesting_contrastive.modules.language_model import LanguageModel
from halvesting_contrastive.utils.helpers import flatten_dict

if TYPE_CHECKING:
    from halvesting_contrastive.utils.configs import BaseConfig


class Retriever(L.LightningModule):

    val_outputs: List[Dict[str, torch.Tensor]]
    test_outputs: List[Dict[str, torch.Tensor]]

    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__()
        flat_params = flatten_dict(cfg.to_dict())
        self.save_hyperparameters(flat_params)
        self.cfg = cfg
        self.lm = LanguageModel(cfg)
        self.contrastive_loss = InfoNCELoss(cfg)

        self.val_ndcg_at_10 = RetrievalNormalizedDCG(top_k=10, sync_on_compute=False)
        self.val_recall_at_10 = RetrievalRecall(top_k=10, sync_on_compute=False)
        self.val_mrr_at_100 = ReciprocalRank(k=100)

        self.test_ndcg_at_10 = RetrievalNormalizedDCG(top_k=10, sync_on_compute=False)
        self.test_recall_at_10 = RetrievalRecall(top_k=10, sync_on_compute=False)
        self.test_mrr_at_100 = ReciprocalRank(k=100)

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

    # --- END HELPER ---

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

    def on_validation_start(self):
        self.val_outputs = []
        self.val_ndcg_at_10.reset()
        self.val_recall_at_10.reset()
        self.val_mrr_at_100.reset()

    def validation_step(self, batch, batch_idx: int) -> None:
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

        # --- MODIFIED: Pool embeddings before storing ---
        q_vec = self._pool_embeddings(q_embs, batch["attention_mask"])
        pos_vec = self._pool_embeddings(pos_embs, batch["pos_attention_mask"])
        neg_vec = self._pool_embeddings(neg_embs, batch["neg_attention_mask"])

        # Store the (much smaller) pooled vectors
        self.val_outputs.append(
            {
                "q_vec": q_vec,
                "pos_vec": pos_vec,
                "neg_vec": neg_vec,
                "index": batch["index"],
            }
        )
        # --- END MODIFICATION ---

    def on_validation_epoch_end(self) -> None:
        # --- MODIFIED: Collate pooled vectors ---
        local_q_vecs = torch.cat([x["q_vec"] for x in self.val_outputs], dim=0)
        local_pos_vecs = torch.cat([x["pos_vec"] for x in self.val_outputs], dim=0)
        local_neg_vecs = torch.cat([x["neg_vec"] for x in self.val_outputs], dim=0)
        local_indices = torch.cat([x["index"] for x in self.val_outputs], dim=0)

        # ALL PROCESSES MUST CALL all_gather
        all_q_vecs = self._gather_and_flatten(local_q_vecs)
        all_pos_vecs = self._gather_and_flatten(local_pos_vecs)
        all_neg_vecs = self._gather_and_flatten(local_neg_vecs)
        all_indices = self._gather_and_flatten(local_indices)
        # --- END MODIFICATION ---

        # Get the full list of targets
        val_targets_list = self.trainer.datamodule.val_targets

        num_gathered_items = len(all_indices)

        # --- UN-SCRAMBLE AND REMOVE PADDING (using torch.unique) ---
        unique_indices, inverse = torch.unique(
            all_indices, sorted=True, return_inverse=True
        )

        if num_gathered_items > len(unique_indices):
            # Padding was present, find the *first* occurrence
            perm = torch.arange(
                inverse.size(0), dtype=inverse.dtype, device=inverse.device
            )
            inverse, perm = inverse.flip([0]), perm.flip([0])
            select_indices = inverse.new_empty(unique_indices.size(0)).scatter_(
                0, inverse, perm
            )
        else:
            # No padding, just sort the scrambled indices
            select_indices = torch.argsort(all_indices)

        query_original_indices = all_indices[select_indices]
        # --- END UN-SCRAMBLE ---

        # --- MODIFIED: Apply selection to pooled vectors ---
        all_q_vecs_sorted = all_q_vecs[select_indices]
        all_pos_vecs_sorted = all_pos_vecs[select_indices]
        all_neg_vecs_sorted = all_neg_vecs[select_indices]

        # Build the GLOBAL corpus from pooled vectors
        corpus_vecs = torch.cat([all_pos_vecs_sorted, all_neg_vecs_sorted], dim=0)

        # Compute scores (chunked) - NO MASKS NEEDED
        scores = self._compute_scores_chunked(all_q_vecs_sorted, corpus_vecs)
        # --- END MODIFICATION ---

        # --- Compute Full Retrieval Metrics (nDCG / Recall) ---
        preds, target, indexes = self._convert_to_torchmetrics_format(
            scores, val_targets_list, query_original_indices
        )

        self.val_ndcg_at_10.update(preds, target, indexes=indexes)
        self.val_recall_at_10.update(preds, target, indexes=indexes)

        # --- Compute Simple MRR (diagonal positive) ---
        num_queries = all_q_vecs_sorted.size(0)
        targets_simple = torch.arange(num_queries, device=self.device)
        self.val_mrr_at_100.update(scores, targets_simple)

        # --- Compute Triplet-Level Accuracy ---
        pos_scores = torch.diag(scores)
        neg_scores = torch.diag(scores[:, num_queries:])
        val_accuracy = (pos_scores > neg_scores).float().mean()

        del all_q_vecs_sorted
        del all_pos_vecs_sorted
        del all_neg_vecs_sorted
        del corpus_vecs
        torch.cuda.empty_cache()

        # --- Log all metrics ---
        # All ranks compute, but only rank 0 logs (sync_dist=False)
        self.log_dict(
            {
                "val/nDCG@10": self.val_ndcg_at_10.compute(),
                "val/Recall@10": self.val_recall_at_10.compute(),
                "val/MRR@100": self.val_mrr_at_100.compute().mean(),
                "val/Accuracy": val_accuracy,
            },
            prog_bar=True,
            sync_dist=False,
        )

        # Clean up (clear lists to free VRAM on ALL ranks)
        self.val_ndcg_at_10.reset()
        self.val_recall_at_10.reset()
        self.val_mrr_at_100.reset()
        self.val_outputs.clear()

    def on_test_start(self) -> None:
        self.test_outputs = []
        # Reset all metrics
        self.test_ndcg_at_10.reset()
        self.test_recall_at_10.reset()
        self.test_mrr_at_100.reset()

    def test_step(self, batch, batch_idx: int) -> None:
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

        # --- MODIFIED: Pool embeddings before storing ---
        q_vec = self._pool_embeddings(q_embs, batch["attention_mask"])
        pos_vec = self._pool_embeddings(pos_embs, batch["pos_attention_mask"])
        neg_vec = self._pool_embeddings(neg_embs, batch["neg_attention_mask"])

        self.test_outputs.append(
            {
                "q_vec": q_vec,
                "pos_vec": pos_vec,
                "neg_vec": neg_vec,
            }
        )
        # --- END MODIFICATION ---

    def on_test_epoch_end(self) -> None:
        # --- SIMPLIFIED TEST LOGIC (SINGLE GPU) ---

        # --- MODIFIED: Collate pooled vectors ---
        local_q_vecs = torch.cat([x["q_vec"] for x in self.test_outputs], dim=0)
        local_pos_vecs = torch.cat([x["pos_vec"] for x in self.test_outputs], dim=0)
        local_neg_vecs = torch.cat([x["neg_vec"] for x in self.test_outputs], dim=0)
        # --- END MODIFICATION ---

        test_targets_list = self.trainer.datamodule.test_targets
        num_processed_queries = local_q_vecs.size(0)

        # No sorting needed for single-GPU test
        all_q_vecs_sorted = local_q_vecs
        all_pos_vecs_sorted = local_pos_vecs
        all_neg_vecs_sorted = local_neg_vecs

        # Build the GLOBAL corpus from pooled vectors
        corpus_vecs = torch.cat([all_pos_vecs_sorted, all_neg_vecs_sorted], dim=0)

        # Compute scores - NO MASKS NEEDED
        scores = self._compute_scores_chunked(all_q_vecs_sorted, corpus_vecs)

        # --- Compute Full Retrieval Metrics (nDCG / Recall) ---
        # Test assumes 1 GPU, shuffle=False, so queries are just [0, 1, ..., N-1]
        query_original_indices = torch.arange(
            num_processed_queries, device=scores.device
        )

        preds, target, indexes = self._convert_to_torchmetrics_format(
            scores, test_targets_list, query_original_indices
        )

        self.test_ndcg_at_10.update(preds, target, indexes=indexes)
        self.test_recall_at_10.update(preds, target, indexes=indexes)

        # --- Compute Simple MRR (diagonal positive) ---
        num_queries = all_q_vecs_sorted.size(0)
        targets_simple = torch.arange(num_queries, device=self.device)
        self.test_mrr_at_100.update(scores, targets_simple)

        # --- Compute Triplet-Level Accuracy ---
        pos_scores = torch.diag(scores)
        neg_scores = torch.diag(scores[:, num_queries:])
        test_accuracy = (pos_scores > neg_scores).float().mean()

        del all_q_vecs_sorted
        del all_pos_vecs_sorted
        del all_neg_vecs_sorted
        del corpus_vecs
        torch.cuda.empty_cache()

        # --- Log all metrics ---
        self.log_dict(
            {
                "test/nDCG@10": self.test_ndcg_at_10.compute(),
                "test/Recall@10": self.test_recall_at_10.compute(),
                "test/MRR@100": self.test_mrr_at_100.compute().mean(),
                "test/Accuracy": test_accuracy,
            },
            prog_bar=True,
            sync_dist=False,  # Not needed for single GPU
        )

        # Clean up (clear lists to free VRAM)
        self.test_ndcg_at_10.reset()
        self.test_recall_at_10.reset()
        self.test_mrr_at_100.reset()
        self.test_outputs.clear()

    def _gather_and_flatten(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gathers a tensor from all DDP processes and flattens the first two
        dims.

        Works for single-GPU (world_size=1) as well.
        """
        # Handle cases where trainer isn't fully initialized or not in DDP
        if not hasattr(self, "trainer") or self.trainer.world_size == 1:
            return tensor

        # all_gather adds a dimension at the start:
        # [num_gpus, local_batch_size, ...]
        gathered_tensor = self.all_gather(tensor)

        # Flatten to: [global_batch_size, ...]
        return gathered_tensor.flatten(0, 1)

    # --- MODIFIED: Accepts pre-pooled vectors ---
    def _compute_scores_chunked(
        self,
        all_q_vecs: Float[torch.Tensor, "num_q hidden"],
        corpus_vecs: Float[torch.Tensor, "num_c hidden"],
        chunk_size=512,
    ) -> Float[torch.Tensor, "num_q num_c"]:
        all_scores_chunks = []
        num_queries = all_q_vecs.size(0)

        for i in tqdm(range(0, num_queries, chunk_size)):
            q_chunk = all_q_vecs[i : i + chunk_size]

            # Embeddings are already pooled, just do the matmul
            chunk_scores = torch.matmul(q_chunk, corpus_vecs.T)

            all_scores_chunks.append(chunk_scores)

        return torch.cat(all_scores_chunks, dim=0)

    # --- END MODIFICATION ---

    @staticmethod
    def _convert_to_torchmetrics_format(
        scores: torch.Tensor,
        targets_list: List[List[int]],
        query_original_indices: torch.Tensor,
    ):
        num_queries, num_corpus_docs = scores.shape
        device = scores.device

        # `preds`: Flat 1D tensor of all scores
        preds = scores.flatten()

        # `indexes`: Flat 1D tensor of query IDs [0, 0, ..., 1, 1, ..., N-1, ...]
        indexes = (
            torch.arange(num_queries, device=device)
            .view(-1, 1)
            .expand(-1, num_corpus_docs)
            .flatten()
        )

        # `target`: Flat 1D boolean tensor
        target_matrix = torch.zeros(
            num_queries, num_corpus_docs, dtype=torch.bool, device=device
        )

        # Use original_indices to pull the correct targets from the full list
        for i, original_idx in enumerate(query_original_indices):

            # Get the correct list of positive doc indices
            relevant_doc_indices = targets_list[original_idx.item()]

            if relevant_doc_indices:
                relevant_doc_indices_tensor = torch.tensor(
                    relevant_doc_indices, dtype=torch.long, device=device
                )
                # Filter out any potential out-of-bounds indices
                valid_indices = relevant_doc_indices_tensor[
                    relevant_doc_indices_tensor < num_corpus_docs
                ]
                if valid_indices.numel() > 0:
                    # Use `i` (0 to num_queries-1) to index the target_matrix row
                    target_matrix[i, valid_indices] = True

        target = target_matrix.flatten()

        return preds, target, indexes
