import os
import platform
from typing import Dict, Optional

from pkg.interface.datasets import IDataPreparationDataset
from pkg.train.datasets import logger
from pkg.utils.other_utils.io import check_and_clean_path


class AbstractDataPreparationDataset(IDataPreparationDataset):
    """Abstract base class for dataset preparation.

    This class handles the preparation of datasets including data generation, statistics calculation,
    and size tracking. It provides a framework for implementing dataset-specific preparation logic.

    Args:
        data_config (Dict): Configuration dictionary containing dataset preparation settings
    """

    def __init__(self, data_config: Dict) -> None:
        logger.info(f"=== Init BaseAbstractDataset {data_config['data_type']} data config start ===")
        logger.info(f"data_config is: {data_config}")

        # common config
        self._platform = platform.system()

        # === data type
        self._data_type = data_config["data_type"]

        # path config
        # === base path
        self._base_repo_path = data_config["repo_path"]
        self._base_data_path = data_config["base_data_path"]

        # === dataset and stats path
        self._stats_data_path = f"{self._base_data_path}/stats/{self._data_type}"
        self._dataset_path = f"{self._base_data_path}/datasets/{self._data_type}"
        self._data_size_path = f"{self._stats_data_path}/data_size.npy"

        logger.info(f"base_repo_path is {self._base_repo_path}")
        logger.info(f"base_data_path is {self._base_data_path}")
        logger.info(f"stats_data_path is {self._stats_data_path}")
        logger.info(f"dataset_path is {self._dataset_path}")
        logger.info(f"data_size_path is {self._data_size_path}")

        # hdf5 config
        self._dataset_h5_path = f"{self._dataset_path}" + "/data_{}.h5"

        logger.info(f"dataset_h5_path is {self._dataset_h5_path}")

        # dataset type
        self._context_description: Optional[Dict[str, str]] = None  # please overwrite this variable
        self._feature_description: Optional[Dict[str, str]] = None  # please overwrite this variable

        # others
        self._overwrite_data = data_config["overwrite_data"]
        self._overwrite_stats = data_config["overwrite_stats"]

        # source
        self._data_config = data_config

    def _validation(self) -> None:
        if not os.path.isdir(self._base_data_path):
            raise NotADirectoryError(f"No directory at: {self._base_data_path}")

    def process(self) -> None:
        """Main dataset preparation process.

        Handles the full dataset preparation workflow:
        1. Generates dataset if needed or requested via overwrite
        2. Calculates statistics for training data
        3. Records total dataset size
        """
        logger.info(f"=== Starting {self._data_type} dataset preparation process ===")

        if check_and_clean_path(self._dataset_path, self._overwrite_data):
            logger.info(f"Generating dataset at: {self._dataset_path}")
            self._data_generation()
        else:
            logger.info(f"Dataset already exists at: {self._dataset_path}")

        if check_and_clean_path(self._stats_data_path, self._overwrite_stats):
            self._data_stats_generation()

        logger.info("=== Dataset preparation process complete ===")

    def _data_generation(self) -> None:
        raise NotImplementedError("please implement 'data_generation' method")

    def _data_stats_generation(self) -> None:
        raise NotImplementedError("please implement 'data_stats_generation' method")

    @property
    def get_config(self) -> Dict:
        config = {
            "platform": self._platform,
            "data_type": self._data_type,
            "base_repo_path": self._base_repo_path,
            "base_data_path": self._base_data_path,
            "stats_data_path": self._stats_data_path,
            "dataset_path": self._dataset_path,
            "data_size_path": self._data_size_path,
            "dataset_h5_path": self._dataset_h5_path,
            "context_description": self._context_description,
            "feature_description": self._feature_description,
            "overwrite_data": self._overwrite_data,
            "overwrite_stats": self._overwrite_stats,
            "source_data_config": self._data_config,
        }

        return config
