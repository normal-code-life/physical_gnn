import argparse
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from common.constant import TEST_NAME, TRAIN_NAME, VALIDATION_NAME
from pkg.utils.data_utils.dataset_generation import split_dataset_indices
from task.passive_biv.data_preparation.datasets_preparation_hdf5 import PassiveBiVPreparationDataset
from task.passive_biv.utils.utils import import_data_config


def load_graph_mapping(mapping_file_path) -> Dict:
    """Load mapping from generated graph to original graph."""
    mapping_df = pd.read_csv(mapping_file_path, names=["original", "generated"])

    return dict(zip(mapping_df["generated"], mapping_df["original"]))


def expand_sample_indices(sample_indices, graph_mapping) -> Dict:
    """Based on graph mapping extend graph samples."""
    sample_indices_extended = {
        TRAIN_NAME: set(sample_indices[TRAIN_NAME]),
        VALIDATION_NAME: set(sample_indices[VALIDATION_NAME]),
    }

    for g_graph, o_graph in graph_mapping.items():
        if o_graph in sample_indices_extended[TRAIN_NAME]:
            sample_indices_extended[TRAIN_NAME].add(g_graph)
        elif o_graph in sample_indices_extended[VALIDATION_NAME]:
            sample_indices_extended[VALIDATION_NAME].add(g_graph)
        else:
            raise ValueError(f"failed to get source graph of {o_graph}")

    return {k: sorted(list(v)) for k, v in sample_indices_extended.items()}


if __name__ == "__main__":
    np.random.seed(753)

    start_time = time.time()

    parser = argparse.ArgumentParser(description="Data Preparation")

    parser.add_argument(
        "--config_path",
        type=str,
        default="task/passive_biv/data_preparation/config/data_config.yaml",
        help="config path location",
    )

    args: (argparse.Namespace, List[str]) = parser.parse_known_args()

    path = args[0].config_path

    data_config = import_data_config(path)

    # generate sample indices
    sample_indices_dict = split_dataset_indices(
        data_config["start_index"],
        data_config["sample_size"],
        data_config["train_split_ratio"],
    )

    print(f"data split: {sample_indices_dict}")

    # read generated to original node map
    graph_mapping = load_graph_mapping(data_config["mapping_file"])

    # generate full sample indices based on the mapping group
    sample_indices_dict = expand_sample_indices(sample_indices_dict, graph_mapping)
    print(f"data expand: {sample_indices_dict}")

    # directly copy validation file to test file
    sample_indices_dict[TEST_NAME] = sample_indices_dict[VALIDATION_NAME]

    # generate dataset
    for data_type in [TRAIN_NAME, VALIDATION_NAME, TEST_NAME]:
        sample_indices = np.array([i - 1 for i in sample_indices_dict[data_type]])  # index - 1

        np.random.shuffle(sample_indices)

        sub_data_config = data_config.copy()
        sub_data_config["data_type"] = data_type
        sub_data_config["sample_indices"] = sample_indices

        data = PassiveBiVPreparationDataset(sub_data_config)

        data.process()

    print(f"data preparation done, total time: {time.time() - start_time}s")
