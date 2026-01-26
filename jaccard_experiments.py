# jaccard_experiments.py

import datasets
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def tokenize(text: str) -> list:
    """A simple tokenizer that lowercases and splits by space."""
    return text.lower().split()


def compute_overlap_metrics(query: str, positive: str, negative: str) -> dict:
    """Compute various overlap metrics between texts using Jaccard
    similarity."""
    q_tokens = set(tokenize(query))
    p_tokens = set(tokenize(positive))
    n_tokens = set(tokenize(negative))

    # Jaccard similarities
    qp_overlap = (
        len(q_tokens & p_tokens) / len(q_tokens | p_tokens)
        if len(q_tokens | p_tokens) > 0
        else 0
    )
    qn_overlap = (
        len(q_tokens & n_tokens) / len(q_tokens | n_tokens)
        if len(q_tokens | n_tokens) > 0
        else 0
    )

    signal = qp_overlap - qn_overlap
    noise = (
        len(p_tokens & n_tokens) / len(p_tokens | n_tokens)
        if len(p_tokens | n_tokens) > 0
        else 0
    )

    return {
        "qp_overlap": qp_overlap,
        "qn_overlap": qn_overlap,
        "signal": signal,
        "noise": noise,
    }


def analyze_vocabulary_patterns(
    dataset_name: str, config: str, sample_size: int = 2000
) -> tuple:
    """Analyze vocabulary patterns in standard vs hard triplets."""
    test_data = datasets.load_dataset(dataset_name, name=config, split="test")

    if len(test_data) > sample_size:
        indices = np.random.choice(len(test_data), sample_size, replace=False)
        test_data = test_data.select(indices)

    standard_metrics = []
    hard_metrics = []

    for example in tqdm(test_data, desc=f"Processing {config}"):
        metrics = compute_overlap_metrics(
            example["query"], example["positive"], example["negative"]
        )

        if example["query_halid"] == example["pos_halid"]:
            standard_metrics.append(metrics)
        else:
            hard_metrics.append(metrics)

    return standard_metrics, hard_metrics


def plot_vocabulary_analysis_for_paper(
    standard_metrics: list, hard_metrics: list, config: str
):
    """Create paper-ready visualization (PDF) without the interpretation
    box."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{config[-1]} sentences", fontsize=16)

    metrics_names = ["qp_overlap", "qn_overlap", "signal", "noise"]
    titles = [
        "Query-Positive Overlap",
        "Query-Negative Overlap",
        "Signal (QP Overlap - QN Overlap)",
        "Noise (PN Overlap)",
    ]

    for idx, (ax, metric, title) in enumerate(zip(axes, metrics_names, titles)):
        standard_vals = [m[metric] for m in standard_metrics]
        hard_vals = [m[metric] for m in hard_metrics]

        ax.hist(
            standard_vals,
            alpha=0.6,
            bins=30,
            label="Unrestricted data",
            color="royalblue",
            density=True,
        )
        ax.hist(
            hard_vals,
            alpha=0.6,
            bins=30,
            label="Base data",
            color="salmon",
            density=True,
        )

        standard_mean = np.mean(standard_vals) if standard_vals else 0
        hard_mean = np.mean(hard_vals) if hard_vals else 0

        ax.axvline(
            standard_mean,
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Unrestricted data mean: {standard_mean:.3f}",
        )
        ax.axvline(
            hard_mean,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Base data mean: {hard_mean:.3f}",
        )

        ax.set_xlabel("Jaccard Similarity")
        if idx == 0:
            ax.set_ylabel("Density")
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save as PDF
    pdf_filename = f"vocabulary_analysis_{config}.pdf"
    plt.savefig(pdf_filename, format="pdf", bbox_inches="tight")
    print(f"Saved paper-ready plot to {pdf_filename}")
    plt.close()


if __name__ == "__main__":
    for i in [2, 4, 6, 8]:
        config_name = f"base-{i}"
        print(f"\n{'='*50}\nAnalyzing {config_name}...")

        standard_m, hard_m = analyze_vocabulary_patterns(
            "almanach/HALvest-Contrastive", config_name, sample_size=10000
        )

        if not standard_m or not hard_m:
            print(f"Skipping plot for {config_name} due to lack of samples.")
            continue

        plot_vocabulary_analysis_for_paper(standard_m, hard_m, config_name)

    print(f"\n{'='*50}\nAnalysis complete. All PDF plots saved to local files.")
