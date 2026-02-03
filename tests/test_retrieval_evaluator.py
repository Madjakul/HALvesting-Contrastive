# tests/test_retrieval_evaluator.py

import pytest
import torch


def test_retrieval_evaluator_imports():
    """Test that RetrievalEvaluator can be imported."""
    from halvesting_contrastive.callbacks import RetrievalEvaluator
    
    evaluator = RetrievalEvaluator()
    assert evaluator is not None
    assert hasattr(evaluator, 'val_outputs')
    assert hasattr(evaluator, 'test_outputs')
    # Verify they are initialized as empty lists
    assert evaluator.val_outputs == []
    assert evaluator.test_outputs == []
    assert hasattr(evaluator, 'val_ndcg_at_10')
    assert hasattr(evaluator, 'val_recall_at_10')
    assert hasattr(evaluator, 'val_mrr_at_100')


def test_retriever_validation_step_returns_dict():
    """Test that Retriever validation_step returns a dictionary with correct keys."""
    from halvesting_contrastive.modules import Retriever
    from halvesting_contrastive.utils.configs import BaseConfig
    
    # Create a minimal config
    cfg = BaseConfig(mode="train")
    
    # Create retriever instance
    retriever = Retriever(cfg)
    
    # Check that validation_step and test_step exist and have correct signatures
    import inspect
    val_sig = inspect.signature(retriever.validation_step)
    test_sig = inspect.signature(retriever.test_step)
    
    # Verify return type annotations if they exist
    assert 'batch' in val_sig.parameters
    assert 'batch_idx' in val_sig.parameters
    assert 'batch' in test_sig.parameters
    assert 'batch_idx' in test_sig.parameters


def test_retriever_removed_methods():
    """Test that evaluation methods were removed from Retriever."""
    from halvesting_contrastive.modules import Retriever
    from halvesting_contrastive.utils.configs import BaseConfig
    
    cfg = BaseConfig(mode="train")
    retriever = Retriever(cfg)
    
    # Verify methods were removed
    assert not hasattr(retriever, 'on_validation_start')
    assert not hasattr(retriever, 'on_validation_epoch_end')
    assert not hasattr(retriever, 'on_test_start')
    assert not hasattr(retriever, 'on_test_epoch_end')
    assert not hasattr(retriever, '_gather_and_flatten')
    assert not hasattr(retriever, '_compute_scores_chunked')
    assert not hasattr(retriever, '_convert_to_torchmetrics_format')
    
    # Verify metrics were removed from __init__
    assert not hasattr(retriever, 'val_ndcg_at_10')
    assert not hasattr(retriever, 'val_recall_at_10')
    assert not hasattr(retriever, 'val_mrr_at_100')
    assert not hasattr(retriever, 'test_ndcg_at_10')
    assert not hasattr(retriever, 'test_recall_at_10')
    assert not hasattr(retriever, 'test_mrr_at_100')
    
    # Verify that essential methods remain
    assert hasattr(retriever, 'training_step')
    assert hasattr(retriever, 'validation_step')
    assert hasattr(retriever, 'test_step')
    assert hasattr(retriever, '_pool_embeddings')


def test_train_utils_has_evaluator_callback():
    """Test that train_utils registers RetrievalEvaluator callback."""
    from halvesting_contrastive.utils import train_utils
    from halvesting_contrastive.callbacks import RetrievalEvaluator
    from halvesting_contrastive.utils.configs import BaseConfig
    
    # Check that RetrievalEvaluator is imported in train_utils
    assert hasattr(train_utils, 'RetrievalEvaluator')
    
    # Verify it's the same class
    assert train_utils.RetrievalEvaluator is RetrievalEvaluator
    
    # Verify that setup_trainer actually registers the callback
    cfg = BaseConfig(mode="train")
    trainer = train_utils.setup_trainer(cfg=cfg, logs_dir="/tmp/logs")
    
    # Check that RetrievalEvaluator is in the callbacks list
    callback_types = [type(cb).__name__ for cb in trainer.callbacks]
    assert 'RetrievalEvaluator' in callback_types, \
        f"RetrievalEvaluator not found in trainer callbacks: {callback_types}"
