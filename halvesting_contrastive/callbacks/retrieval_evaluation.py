# halvesting_contrastive/callbacks/retrieval_evaluation.py

import logging
from typing import Any, List

import lightning as L
import torch
from jaxtyping import Float
from torcheval.metrics import ReciprocalRank
from torchmetrics.retrieval import RetrievalNormalizedDCG, RetrievalRecall
from tqdm import tqdm


class RetrievalEvaluator(L.Callback):
    """Callback to handle retrieval evaluation metrics for validation and test."""

    def __init__(self):
        super().__init__()
        self.val_outputs: List[dict] = []
        self.test_outputs: List[dict] = []
        
        # Validation metrics
        self.val_ndcg_at_10 = RetrievalNormalizedDCG(top_k=10, sync_on_compute=False)
        self.val_recall_at_10 = RetrievalRecall(top_k=10, sync_on_compute=False)
        self.val_mrr_at_100 = ReciprocalRank(k=100)
        
        # Test metrics
        self.test_ndcg_at_10 = RetrievalNormalizedDCG(top_k=10, sync_on_compute=False)
        self.test_recall_at_10 = RetrievalRecall(top_k=10, sync_on_compute=False)
        self.test_mrr_at_100 = ReciprocalRank(k=100)

    def on_validation_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Reset outputs and metrics at the start of validation."""
        self.val_outputs = []
        self.val_ndcg_at_10.reset()
        self.val_recall_at_10.reset()
        self.val_mrr_at_100.reset()

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect outputs from each validation batch."""
        if outputs is not None:
            self.val_outputs.append(outputs)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Compute and log validation metrics at the end of the epoch."""
        if not self.val_outputs:
            return

        local_q_vecs = torch.cat([x["q_vec"] for x in self.val_outputs], dim=0)
        local_pos_vecs = torch.cat([x["pos_vec"] for x in self.val_outputs], dim=0)
        local_neg_vecs = torch.cat([x["neg_vec"] for x in self.val_outputs], dim=0)
        local_indices = torch.cat([x["index"] for x in self.val_outputs], dim=0)

        all_q_vecs = self._gather_and_flatten(local_q_vecs, trainer)
        all_pos_vecs = self._gather_and_flatten(local_pos_vecs, trainer)
        all_neg_vecs = self._gather_and_flatten(local_neg_vecs, trainer)
        all_indices = self._gather_and_flatten(local_indices, trainer)

        val_targets_list = trainer.datamodule.val_targets

        num_gathered_items = len(all_indices)

        unique_indices, inverse = torch.unique(
            all_indices, sorted=True, return_inverse=True
        )

        if num_gathered_items > len(unique_indices):
            perm = torch.arange(
                inverse.size(0), dtype=inverse.dtype, device=inverse.device
            )
            inverse, perm = inverse.flip([0]), perm.flip([0])
            select_indices = inverse.new_empty(unique_indices.size(0)).scatter_(
                0, inverse, perm
            )
        else:
            select_indices = torch.argsort(all_indices)

        query_original_indices = all_indices[select_indices]

        all_q_vecs_sorted = all_q_vecs[select_indices]
        all_pos_vecs_sorted = all_pos_vecs[select_indices]
        all_neg_vecs_sorted = all_neg_vecs[select_indices]

        corpus_vecs = torch.cat([all_pos_vecs_sorted, all_neg_vecs_sorted], dim=0)

        scores = self._compute_scores_chunked(all_q_vecs_sorted, corpus_vecs)

        preds, target, indexes = self._convert_to_torchmetrics_format(
            scores, val_targets_list, query_original_indices
        )

        self.val_ndcg_at_10.update(preds, target, indexes=indexes)
        self.val_recall_at_10.update(preds, target, indexes=indexes)

        num_queries = all_q_vecs_sorted.size(0)
        targets_simple = torch.arange(num_queries, device=pl_module.device)
        self.val_mrr_at_100.update(scores, targets_simple)

        pos_scores = torch.diag(scores)
        neg_scores = torch.diag(scores[:, num_queries:])
        val_accuracy = (pos_scores > neg_scores).float().mean()

        del all_q_vecs_sorted
        del all_pos_vecs_sorted
        del all_neg_vecs_sorted
        del corpus_vecs
        torch.cuda.empty_cache()

        pl_module.log_dict(
            {
                "val/nDCG@10": self.val_ndcg_at_10.compute(),
                "val/Recall@10": self.val_recall_at_10.compute(),
                "val/MRR@100": self.val_mrr_at_100.compute().mean(),
                "val/Accuracy": val_accuracy,
            },
            prog_bar=True,
            sync_dist=False,
        )

        self.val_ndcg_at_10.reset()
        self.val_recall_at_10.reset()
        self.val_mrr_at_100.reset()
        self.val_outputs.clear()

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Reset outputs and metrics at the start of testing."""
        self.test_outputs = []
        self.test_ndcg_at_10.reset()
        self.test_recall_at_10.reset()
        self.test_mrr_at_100.reset()

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect outputs from each test batch."""
        if outputs is not None:
            self.test_outputs.append(outputs)

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Compute and log test metrics at the end of the epoch."""
        if not self.test_outputs:
            return

        local_q_vecs = torch.cat([x["q_vec"] for x in self.test_outputs], dim=0)
        local_pos_vecs = torch.cat([x["pos_vec"] for x in self.test_outputs], dim=0)
        local_neg_vecs = torch.cat([x["neg_vec"] for x in self.test_outputs], dim=0)

        test_targets_list = trainer.datamodule.test_targets
        num_processed_queries = local_q_vecs.size(0)

        all_q_vecs_sorted = local_q_vecs
        all_pos_vecs_sorted = local_pos_vecs
        all_neg_vecs_sorted = local_neg_vecs

        corpus_vecs = torch.cat([all_pos_vecs_sorted, all_neg_vecs_sorted], dim=0)

        scores = self._compute_scores_chunked(all_q_vecs_sorted, corpus_vecs)

        # Test assumes 1 GPU, shuffle=False, so queries are just [0, 1, ..., N-1]
        query_original_indices = torch.arange(
            num_processed_queries, device=scores.device
        )

        preds, target, indexes = self._convert_to_torchmetrics_format(
            scores, test_targets_list, query_original_indices
        )

        self.test_ndcg_at_10.update(preds, target, indexes=indexes)
        self.test_recall_at_10.update(preds, target, indexes=indexes)

        num_queries = all_q_vecs_sorted.size(0)
        targets_simple = torch.arange(num_queries, device=pl_module.device)
        self.test_mrr_at_100.update(scores, targets_simple)

        pos_scores = torch.diag(scores)
        neg_scores = torch.diag(scores[:, num_queries:])
        test_accuracy = (pos_scores > neg_scores).float().mean()

        del all_q_vecs_sorted
        del all_pos_vecs_sorted
        del all_neg_vecs_sorted
        del corpus_vecs
        torch.cuda.empty_cache()

        pl_module.log_dict(
            {
                "test/nDCG@10": self.test_ndcg_at_10.compute(),
                "test/Recall@10": self.test_recall_at_10.compute(),
                "test/MRR@100": self.test_mrr_at_100.compute().mean(),
                "test/Accuracy": test_accuracy,
            },
            prog_bar=True,
            sync_dist=False,  # Not needed for single GPU
        )

        self.test_ndcg_at_10.reset()
        self.test_recall_at_10.reset()
        self.test_mrr_at_100.reset()
        self.test_outputs.clear()

    def _gather_and_flatten(
        self, tensor: torch.Tensor, trainer: L.Trainer
    ) -> torch.Tensor:
        """Gathers a tensor from all DDP processes and flattens the first two dims.

        Works for single-GPU (world_size=1) as well.
        """
        if trainer.world_size == 1:
            return tensor

        # Check if distributed is initialized before using all_gather
        if not torch.distributed.is_initialized():
            return tensor

        # all_gather adds a dimension at the start:
        # [num_gpus, local_batch_size, ...]
        gathered_list = [torch.zeros_like(tensor) for _ in range(trainer.world_size)]
        torch.distributed.all_gather(gathered_list, tensor)
        gathered_tensor = torch.stack(gathered_list, dim=0)

        # Flatten to: [global_batch_size, ...]
        return gathered_tensor.flatten(0, 1)

    def _compute_scores_chunked(
        self,
        all_q_vecs: Float[torch.Tensor, "num_q hidden"],
        corpus_vecs: Float[torch.Tensor, "num_c hidden"],
        chunk_size=512,
    ) -> Float[torch.Tensor, "num_q num_c"]:
        """Compute dot product scores in chunks to avoid memory issues."""
        all_scores_chunks = []
        num_queries = all_q_vecs.size(0)

        for i in tqdm(range(0, num_queries, chunk_size)):
            q_chunk = all_q_vecs[i : i + chunk_size]

            # Embeddings are already pooled, just do the matmul
            chunk_scores = torch.matmul(q_chunk, corpus_vecs.T)

            all_scores_chunks.append(chunk_scores)

        return torch.cat(all_scores_chunks, dim=0)

    @staticmethod
    def _convert_to_torchmetrics_format(
        scores: torch.Tensor,
        targets_list: List[List[int]],
        query_original_indices: torch.Tensor,
    ):
        """Convert scores to torchmetrics format."""
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
