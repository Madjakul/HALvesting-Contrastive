# contrastive_experiments.py

import logging
import os
from pathlib import Path

import datasets
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    TripletEvaluator,
)

from halvesting_contrastive.utils.logger import logging_config

logging_config()


YOUR_HF_USERNAME = "Madjakul"
YOUR_DATASET_NAME = "HALvest-Contrastive"
BASE_MODEL_NAME = "roberta-base"
NUM_EPOCHS = 1
TRAIN_BATCH_SIZE = 64  # per device
EVALUATION_STEPS = 1000

os.environ["WANDB_PROJECT"] = "authorship"


def evaluate_on_beir(model_path_or_name: str, name=BASE_MODEL_NAME):
    logging.info(
        f"--- Evaluating model '{Path(model_path_or_name).name}' on SciFact ---"
    )
    dataset_path = util.download_and_unzip(
        f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        "datasets",
    )
    Path(f"output/beir-results/{name}").mkdir(parents=True, exist_ok=True)
    corpus, queries, qrels = GenericDataLoader(data_folder=dataset_path).load(
        split="test"
    )

    model = SentenceTransformer(model_path_or_name)
    evaluator = InformationRetrievalEvaluator(
        queries, corpus, qrels, name="scifact-test"
    )
    results = evaluator(model, output_path=f"output/beir-results/{Path(name).name}")
    logging.info(f"SciFact evaluation complete. Full results in output folder.")


def evaluate_model_on_base_splits(model_path: str, output_dir: str, hard=False):
    logging.info(
        f"--- Evaluating model '{Path(model_path).name}' on base test splits ---"
    )
    model = SentenceTransformer(model_path)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for i in [2, 4, 6, 8]:
        logging.info(f"--- Loading base-{i} test split ---")
        test_data = datasets.load_dataset(
            f"{YOUR_HF_USERNAME}/{YOUR_DATASET_NAME}", name=f"base-{i}", split="test"
        )
        if hard:
            test_data = test_data.filter(
                lambda ex: ex["query_halid"] != ex["pos_halid"]
            )

        evaluator = TripletEvaluator(
            anchors=test_data["query"],
            positives=test_data["positive"],
            negatives=test_data["negative"],
            name=f"base-{i}-test",
        )

        scores = evaluator(model, output_path=output_dir)


def run_experiment_A_standard_verification(base_train_data, base_val_data):
    logging.info(
        "========== Starting Experiment A: Standard Verification Model =========="
    )

    train_dataset = base_train_data.select_columns(["query", "positive", "negative"])
    eval_dataset = base_val_data.select_columns(["query", "positive", "negative"])

    model = SentenceTransformer(BASE_MODEL_NAME)
    loss = losses.TripletLoss(model=model)

    evaluator = TripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name="val",
    )

    args = SentenceTransformerTrainingArguments(
        output_dir="output/exp-A-verification-model",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=TRAIN_BATCH_SIZE,
        eval_strategy="steps",
        eval_steps=EVALUATION_STEPS,
        save_strategy="steps",
        save_steps=EVALUATION_STEPS,
        logging_steps=25,
        fp16=True,
        run_name="exp-A-standard-verification",
        report_to="wandb",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()


# ---


def run_experiment_B_decoupled_verification(base_train_data, base_val_data):
    logging.info(
        "========== Starting Experiment B: Topic-Decoupled Verification Model =========="
    )
    decorrelated_train_data = base_train_data.filter(
        lambda ex: ex["query_halid"] != ex["pos_halid"]
    )
    logging.info(f"De-correlated dataset size: {len(decorrelated_train_data)}")

    train_dataset = decorrelated_train_data.select_columns(
        ["query", "positive", "negative"]
    )
    eval_dataset = base_val_data.select_columns(["query", "positive", "negative"])

    model = SentenceTransformer(BASE_MODEL_NAME)
    loss = losses.TripletLoss(model=model)

    evaluator = TripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name="val",
    )

    args = SentenceTransformerTrainingArguments(
        output_dir="output/exp-B-decoupled-model",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        eval_strategy="steps",
        eval_steps=EVALUATION_STEPS,
        save_strategy="steps",
        save_steps=EVALUATION_STEPS,
        logging_steps=25,
        fp16=True,
        run_name="exp-B-decoupled-verification",
        report_to="wandb",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()


# ---


def run_experiment_C_ict_model(ict_train_data, ict_val_data):
    logging.info("========== Starting Experiment C: ICT Retrieval Model ==========")

    train_dataset = ict_train_data.select_columns(["query", "positive"])

    model = SentenceTransformer(BASE_MODEL_NAME)
    loss = losses.MultipleNegativesRankingLoss(model=model)

    logging.info("Preparing validation data for IR evaluation...")
    queries = {}
    corpus = {}
    qrels = {}

    unique_positives = set(ict_val_data["positive"])

    for idx, passage in enumerate(unique_positives):
        passage_id = f"p{idx}"
        corpus[passage_id] = passage

    passage_to_id = {passage: id for id, passage in corpus.items()}

    # Create queries and qrels
    for idx, example in enumerate(ict_val_data):
        query_text = example["query"]
        positive_passage_text = example["positive"]

        query_id = f"q{idx}"
        queries[query_id] = query_text

        positive_passage_id = passage_to_id[positive_passage_text]

        if query_id not in qrels:
            qrels[query_id] = set()
        qrels[query_id].add(positive_passage_id)

    evaluator = InformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=qrels, name="val"
    )
    args = SentenceTransformerTrainingArguments(
        output_dir="output/exp-C-ict-model",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        eval_strategy="steps",
        eval_steps=EVALUATION_STEPS,
        save_strategy="steps",
        save_steps=EVALUATION_STEPS,
        logging_steps=25,
        fp16=True,
        run_name="exp-C-ict-retrieval",
        report_to="wandb",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()


if __name__ == "__main__":
    # --- Load and prepare all datasets first ---
    # (Your existing dataset loading code is fine, keep it here)
    # base_train_list = [
    #     datasets.load_dataset(
    #         f"{YOUR_HF_USERNAME}/{YOUR_DATASET_NAME}", name=f"base-{i}", split="train"
    #     )
    #     for i in [2, 4, 6, 8]
    # ]
    # base_train = datasets.concatenate_datasets(base_train_list)
    # base_val_list = [
    #     datasets.load_dataset(
    #         f"{YOUR_HF_USERNAME}/{YOUR_DATASET_NAME}", name=f"base-{i}", split="valid"
    #     )
    #     for i in [2, 4, 6, 8]
    # ]
    # base_val = datasets.concatenate_datasets(base_val_list)

    # ict_train_list = [
    #     datasets.load_dataset(
    #         f"{YOUR_HF_USERNAME}/{YOUR_DATASET_NAME}", name=f"ict-{i}", split="train"
    #     )
    #     for i in [1, 2, 3, 4]
    # ]
    # ict_train = datasets.concatenate_datasets(ict_train_list)
    # ict_val_list = [
    #     datasets.load_dataset(
    #         f"{YOUR_HF_USERNAME}/{YOUR_DATASET_NAME}", name=f"ict-{i}", split="valid"
    #     )
    #     for i in [1, 2, 3, 4]
    # ]
    # ict_val = datasets.concatenate_datasets(ict_val_list)

    # --- Run Training Experiments ---
    # logging.info("========== Starting Training Runs ==========")
    # run_experiment_A_standard_verification(base_train, base_val)
    # run_experiment_B_decoupled_verification(base_train, base_val)
    # run_experiment_C_ict_model(ict_train, ict_val)

    # --- Run Final Evaluations on test sets ---
    # logging.info("========== Running Final Evaluations ==========")

    final_checkpoint_A = "checkpoint-3000"  # Change XXXX to your last checkpoint number
    final_checkpoint_B = "checkpoint-3000"  # Change XXXX to your last checkpoint number

    # 1. Evaluate Exp A model on individual base test splits
    evaluate_model_on_base_splits(
        f"output/exp-A-verification-model/{final_checkpoint_A}",
        "output/exp-A-verification-model/test-results",
        hard=True,
    )

    # 2. Evaluate Exp B model on individual base test splits
    evaluate_model_on_base_splits(
        f"output/exp-B-decoupled-model/{final_checkpoint_B}",
        "output/exp-B-decoupled-model/test-results",
        hard=True,
    )

    # # 4. Evaluate Exp C model on the full base test set
    # logging.info("--- Evaluating ICT model on the full base test set ---")
    # final_checkpoint_C = "checkpoint-3495"
    # evaluate_model_on_base_splits(
    #     f"output/exp-C-ict-model/{final_checkpoint_C}",
    #     "output/exp-C-ict-model/base-test-results",
    #     hard=True,
    # )

    # 5. Evaluate baseline on SciFact
    # evaluate_on_beir(BASE_MODEL_NAME)
    # evaluate_on_beir(
    #     f"output/exp-C-ict-model/{final_checkpoint_C}", name=final_checkpoint_C
    # )
    #
    # logging.info("========== All experiments complete! ==========")
