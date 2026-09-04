"""Configuration path resolution for passive BiV data preparation."""

import os
import sys
from typing import Dict

from pkg.utils.other_utils.io import get_repo_path, load_yaml


def import_data_config(file_path: str) -> Dict:
    """Import and merge data configuration from yaml files.

    Args:
        file_path (str): Path to the yaml configuration file, relative to repo root

    Returns:
        Dict: Configuration dictionary containing data generation config settings for downstream use

    Loads the base data configuration from a yaml file and merges it with task-specific settings.
    Generates standard paths based on the repository structure.
    """
    # Get absolute path of current execution file
    cur_path = os.path.abspath(sys.argv[0])

    # Get repository root directory path
    repo_root_path = get_repo_path(cur_path)

    # Load data configuration from yaml file
    data_config = load_yaml(f"{repo_root_path}/{file_path}")

    # Add additional path configurations
    # repo_path: Repository root directory
    data_config["repo_path"] = repo_root_path

    # base_data_path: Root directory for task-specific data
    data_config["base_data_path"] = f"{repo_root_path}/data/{data_config['dataset_name']}"

    # mapping_file: Path to generated-to-original graph mapping file
    data_config["mapping_file"] = f"{data_config['base_data_path']}/generated_to_original.csv"

    return data_config
