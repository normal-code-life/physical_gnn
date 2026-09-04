"""Legacy base dataset with shared paths and hardware configuration."""

import abc
import os
import platform
from typing import Dict, Optional

from pkg.train.datasets import logger


class BaseAbstractDataset(abc.ABC):
    """Base abstract class for dataset handling.

    Provides core functionality for dataset preparation and training, including path setup
    and hardware configuration. Subclasses implement specific data processing logic.

    Attributes:
        base_data_path (str): Base data directory path
        base_task_path (str): Base task directory path
        gpu (bool): Whether to use GPU
        cuda_core (str): CUDA core identifier
        data_type (str): Dataset type (e.g. train, test)
        exp_name (str, optional): Experiment name
    """

    def __init__(self, data_config: Dict, data_type: str, *args, **kwargs) -> None:
        """Initialize dataset with config.

        Args:
            data_config (Dict): Configuration for paths and hardware
            data_type (str): Dataset type (e.g. train, test)
            *args: Additional args
            **kwargs: Additional kwargs
        """
        logger.info(f"=== Init BaseAbstractDataset {data_type} data config start ===")
        logger.info(f"data_config is: {data_config}")

        # Capture hardware configuration for data-loading decisions.
        self.gpu = data_config.get("gpu", False)
        self.cuda_core = data_config.get("cuda_core", 0)
        self.platform = platform.system()

        # Identify the model and task that consume this dataset.
        self.model_name = data_config["model_name"]

        self.task_name = data_config["task_name"]
        self.data_type = data_type

        # The experiment name is optional for reusable datasets.
        self.exp_name = data_config.get("exp_name", "")

        # Resolve the repository, data, and task roots.
        self.base_repo_path = data_config["repo_path"]
        self.base_data_path = data_config["task_data_path"]
        self.base_task_path = data_config["task_path"]

        if not os.path.isdir(self.base_data_path):
            raise NotADirectoryError(f"No directory at: {self.base_data_path}")

        # Derive paths for conventional file-based training data.
        self.stats_data_path = f"{self.base_data_path}/stats"
        self.dataset_path = f"{self.base_data_path}/datasets/{self.data_type}"
        self.data_size_path = f"{self.stats_data_path}/{self.data_type}_data_size.npy"

        logger.info(f"base_data_path is {self.base_data_path}")
        logger.info(f"base_task_path is {self.base_task_path}")
        logger.info(f"stats_data_path is {self.stats_data_path}")
        logger.info(f"dataset_path is {self.dataset_path}")
        logger.info(f"data_size_path is {self.data_size_path}")

        # HDF5 shards are addressed by a numeric format placeholder.
        self.dataset_h5_path = f"{self.dataset_path}" + "/data_{}.h5"

        logger.info(f"dataset_h5_path is {self.dataset_h5_path}")

        # TFRecord shards follow the same numeric naming convention.
        self.tfrecord_data_path = f"{self.dataset_path}" + "/data_{}.tfrecord"

        logger.info(f"tfrecord_data_path is {self.tfrecord_data_path}")

        # No TFRecord compression is assumed by default.
        self.compression_type = None

        # Concrete datasets must supply schemas for context and sequence data.
        self.context_description: Optional[Dict[str, str]] = None

        self.feature_description: Optional[Dict[str, str]] = None

        logger.info(f"=== Init BaseAbstractDataset {data_type} data config done ===")
