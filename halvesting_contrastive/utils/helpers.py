# halvesting_contrastive/utils/helpers.py

import gzip
import json
import logging
import os
import os.path as osp
import shutil
from typing import List

import pandas as pd
import yaml
from tqdm import tqdm

WIDTH = 88

def load_config_from_file(path: str):
    """Load a configuration file from a path.

    Parameters
    ----------
    path: str
        Path to the configuration file.

    Returns
    -------
    config: Dict[str, Any]
        Configuration dictionary.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    load_config_map = {
        "json": load_config_from_json,
        "yaml": load_config_from_yaml,
        "yml": load_config_from_yaml,
    }
    assert os.path.isfile(path)
    _, file_extension = osp.splitext(path)
    file_extension = file_extension[1:]
    if file_extension not in load_config_map:
        raise ValueError(f"File extension {file_extension} not supported.")
    config = load_config_map[file_extension](path)
    return config


def load_config_from_json(path: str):
    """Load a JSON configuration file from a path.

    Parameters
    ----------
    path: str
        Path to the JSON configuration file.

    Returns
    -------
    config: Dict[str, Any]
        Configuration dictionary.
    """
    config = json.load(open(path))
    return config


def load_config_from_yaml(path: str):
    """Load a YAML configuration file from a path.

    Parameters
    ----------
    path: str
        Path to the YAML configuration file.

    Returns
    -------
    config: Dict[str, Any]
        Configuration dictionary.
    """
    config = yaml.load(open(path), Loader=yaml.FullLoader)
    return config


def check_dir(path: str):
    """Check if there is a directory at ``path`` and creates it if necessary.

    Parameters
    ----------
    path: str
        Path to the directory.

    Returns
    -------
    path: str
        Path to the existing directory.
    """
    if os.path.isdir(path):
        return path
    logging.warning(f"No folder at {path}: creating folders at path.")
    os.makedirs(path)
    return path


def read_json(path: str):
    """Reads a JSON file at path.

    Parameters
    ----------

    Returns
    -------
    js: List[Dict[str, Union[str, List[str]]]]
    """
    with open(path, "r", encoding="utf-8") as jsf:
        js = json.load(jsf)
    return js


def read_jsons(paths: List[str]):
    """Reads a JSON files in the directory at path.

    Parameters
    ----------

    Returns
    -------
    js: List[Dict[str, Union[str, List[str]]]]
    """
    js = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as jsf:
            js.extend(json.load(jsf))
    return js


def json_to_dict(path: str, on: str):
    """Convert a JSON file to a dictionary.

    Parameters
    ----------
    path : str
        Path to the JSON file.
    on : str
        Key to use as the main key in the dictionary.

    Returns
    -------
    data : dict
        Dictionary with the JSON data.
    """
    data = {}
    with open(path, "r", encoding="utf-8") as jsf:
        json_headers = json.load(jsf)
    for json_header in json_headers:
        tmp_data = {}
        tmp_data[json_header[on]] = {k: v for k, v in json_header.items() if k != on}
        data.update(tmp_data)
    return data


def jsons_to_dict(paths: List[str], on: str):
    """Convert a list of JSON files to a dictionary.

    Parameters
    ----------
    paths : List[str]
        List of paths to JSON files.
    on : str
        Key to use as the main key in the dictionary.

    Returns
    -------
    data : dict
        Dictionary with the JSON data.
    """
    logging.info("Converting JSONs to Dict...")
    data = {}
    for path in tqdm(paths):
        with open(path, "r", encoding="utf-8") as jsf:
            json_headers = json.load(jsf)
        for json_header in json_headers:
            tmp_data = {}
            tmp_data[json_header[on]] = {
                k: v for k, v in json_header.items() if k != on
            }
            data.update(tmp_data)
    return data


def gzip_compress(path: str):
    """Compress a file to a gzip file.

    Parameters
    ----------
    path : str
        Path to the file to compress.
    """
    with open(path, "rb") as f:
        with gzip.open(f"{path}.gz", "wb") as gzf:
            gzf.writelines(f)


def zip_compress(path: str):
    """Compress a file to a zip file.

    Parameters
    ----------
    path : str
        Path to the file to compress.
    """
    shutil.make_archive(path, "zip", path)


def jsons_to_jsonls(input_paths: List[str], output_paths: List[str]):
    """Convert a list of JSON files to JSONL files.

    Parameters
    ----------
    input_paths : List[str]
        List of paths to JSON files.
    output_paths : List[str]
        List of paths to JSONL files.
    """
    logging.info("Converting JSONs to JSONLs...")
    for input_path, output_path in tqdm(list(zip(input_paths, output_paths))):
        df = pd.read_json(input_path, orient="records")
        df.to_json(output_path, orient="records", lines=True, force_ascii=False)

