# HALvesting-Contrastive

Mining sentence pairs from HAL for contrastive learning and stylometric analysis.

[![arXiv](https://img.shields.io/badge/arXiv-2407.20595-b31b1b.svg)](https://arxiv.org/abs/2407.20595)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Data-yellow)](https://huggingface.co/datasets/Madjakul/HALvest-Contrastive)

See also:

- [HALvesting](https://github.com/Madjakul/HALvesting/)
- [HALvesting-Geometric](https://github.com/Madjakul/HALvesting-Geometric)

---

## Features

- [`preprocess.py`](preprocess.py): Preprocesses the raw data from HAL alongside responses from the API.
- [`sample_pairs.py`](sample_pairs.py): Samples contrastive pairs from the preprocessed data.
- [`postprocess.py`](postprocess.py): Postprocesses the sampled pairs to create the final dataset.

## Requirements

- Python > 3.8.
- A HuggingFace account and being authenticated on your current machine.

## Installation

1. Clone the repository

```bash
git clone https://github.com/Madjakul/HALvesting-Contrastive
```

2. Navigate to the repository

```bash
cd HALvesting-Contrastive
```

3. Install the requirements

```bash
pip install -r requirements.txt
```

## Usage

It's easier to modify the files in [`scripts/`](./scripts/) at need before directly launching them.
You might need to modify the rights of the scripts to make them executable:

```bash
chmod +x scripts/*
```

Don't forget to also modify the configuration files in [`configs/`](./configs/) to suit your needs.

### Preprocessing

Preprocess the raw data from HAL alongside responses from the API.

```
usage: preprocess.py [-h] --config_path CONFIG_PATH [--responses_dir RESPONSES_DIR] [--metadata_path METADATA_PATH]
                     [--num_proc NUM_PROC]

Argument parser used to preprocess HALvest's documents.

options:
  -h, --help            show this help message and exit
  --config_path CONFIG_PATH
                        Path to the config file.
  --responses_dir RESPONSES_DIR
                        Path to the responses directory. Only used if do_convert_responses is True.
  --metadata_path METADATA_PATH
                        Path to the metadata file. If do_convert_responses is True, the metadata will be saved here.
  --num_proc NUM_PROC   Number of processes to use. Default is the number of CPUs.
```

### Sampling

Sample contrastive pairs from the preprocessed data. The pairs can be independent sentences or contiguous ones in order to perform inverse cloze tasks.

```
usage: sample_pairs.py [-h] --config_path CONFIG_PATH [--num_proc NUM_PROC] [--cache_dir CACHE_DIR]

Argument parser used to sample sentences, paragraphs or inverse cloze pairs.

options:
  -h, --help            show this help message and exit
  --config_path CONFIG_PATH
                        Path to the config file.
  --num_proc NUM_PROC   Number of processes to use. Default is the number of CPUs.
  --cache_dir CACHE_DIR
                        Path to the cache directory.
```

### Postprocessing

Postprocess the sampled pairs to create the final dataset.

```
usage: postprocess.py [-h] --config_path CONFIG_PATH [--num_proc NUM_PROC]

Argument parser used to clean out sentence pairs.

options:
  -h, --help            show this help message and exit
  --config_path CONFIG_PATH
                        Path to the config file.
  --num_proc NUM_PROC   Number of processes to use. Default is the number of CPUs.
```

## Citation

To cite this work or the subsequent dataset, please use the following BibTeX entry:

```bib
@misc{kulumba2024harvestingtextualstructureddata,
      title={Harvesting Textual and Structured Data from the HAL Publication Repository},
      author={Francis Kulumba and Wissam Antoun and Guillaume Vimont and Laurent Romary},
      year={2024},
      eprint={2407.20595},
      archivePrefix={arXiv},
      primaryClass={cs.DL},
      url={https://arxiv.org/abs/2407.20595},
}
```
