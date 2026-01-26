# bm25_experiments.py

import datasets
from rank_bm25 import BM25Okapi
from tqdm import tqdm


def tokenize(text: str) -> list:
    """A simple tokenizer that lowercases and splits by space."""
    return text.lower().split()


def evaluate_bm25_on_base_splits_fast(dataset_name: str, hard_only: bool = False):
    """Evaluates BM25 using a robust background corpus but with a much faster
    scoring loop."""
    results = {}

    for i in [2, 4, 6, 8]:
        config_name = f"base-{i}"
        print(f"\n--- Evaluating BM25 on {config_name} (Fast Method) ---")
        test_data = datasets.load_dataset(dataset_name, name=config_name, split="test")

        if hard_only:
            test_data = test_data.filter(
                lambda ex: ex["query_halid"] != ex["pos_halid"]
            )

        print("Building background corpus for global statistics...")
        all_docs = list(
            set(
                [ex["positive"] for ex in test_data]
                + [ex["negative"] for ex in test_data]
            )
        )
        tokenized_corpus = [
            tokenize(doc) for doc in tqdm(all_docs, desc="Tokenizing corpus")
        ]

        print("Initializing master BM25 model...")
        master_bm25 = BM25Okapi(tokenized_corpus)

        correct = 0
        total = len(test_data)

        print("Scoring triplets...")
        for example in tqdm(test_data, desc=f"Evaluating {config_name}"):
            query = example["query"]
            positive_doc = example["positive"]
            negative_doc = example["negative"]
            tokenized_query = tokenize(query)

            # Create a temporary, tiny corpus for this specific triplet
            temp_corpus_docs = [positive_doc, negative_doc]
            tokenized_temp_corpus = [tokenize(doc) for doc in temp_corpus_docs]

            # Initialize a temporary BM25 model
            temp_bm25 = BM25Okapi(tokenized_temp_corpus)

            # *** The Optimization Step ***
            # Inject the global statistics from the master model
            temp_bm25.idf = master_bm25.idf
            temp_bm25.avgdl = master_bm25.avgdl

            # Now, score the query against only the two relevant documents
            scores = temp_bm25.get_scores(tokenized_query)

            if scores[0] > scores[1]:  # scores[0] is positive, scores[1] is negative
                correct += 1

        accuracy = correct / total
        eval_type = "hard" if hard_only else "all"
        print(f"BM25 Accuracy on {config_name} ({eval_type}): {accuracy:.4f}")
        results[config_name] = accuracy

    return results


if __name__ == "__main__":
    YOUR_DATASET_NAME = "Madjakul/HALvest-Contrastive"

    print("\n" + "=" * 50)
    print("Running evaluation with ROBUST IDF (Fast Method)")
    print("=" * 50)

    print("\n--- Evaluating on ALL test sets ---")
    results_all_fast = evaluate_bm25_on_base_splits_fast(
        YOUR_DATASET_NAME, hard_only=False
    )

    print("\n--- Evaluating on HARD test sets ---")
    results_hard_fast = evaluate_bm25_on_base_splits_fast(
        YOUR_DATASET_NAME, hard_only=True
    )

    print("\n\n" + "=" * 20 + " SUMMARY (Fast Method) " + "=" * 20)
    print("All test set:")
    for k, v in results_all_fast.items():
        print(f"{k}: {v:.4f}")
    print("\nHard test set:")
    for k, v in results_hard_fast.items():
        print(f"{k}: {v:.4f}")
