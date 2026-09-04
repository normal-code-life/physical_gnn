"""Indexed and iterable base datasets for model training."""

import abc
import os
from typing import Dict, Optional

import numpy as np
import tfrecord
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from torchvision import transforms

from common.constant import TRAIN_NAME
from pkg.interface.datasets import ITrainDataset
from pkg.train.datasets import logger
from pkg.utils.data_utils.reader_hdf5 import multi_hdf5_loader, shuffle_iterator


class BaseTrainDataset(ITrainDataset):
    """Base train class for dataset handling.

    Provides core functionality for dataset preparation and training, including path setup
    and hardware configuration. Subclasses implement specific data processing logic.

    Args:
        data_config (Dict): Configuration dictionary containing dataset preparation settings
    """

    def __init__(self, data_config: Dict, *args, **kwargs) -> None:
        """Initialize dataset with config.

        Args:
            data_config (Dict): Configuration for paths and hardware
            *args: Additional args
            **kwargs: Additional kwargs
        """
        logger.info(f"=== Init {data_config['data_type']} data config start ===")
        logger.info(f"data_config is: {data_config}")

        # Model and task names were historically stored here; callers now own them.
        # self._model_name = data_config["model_name"]

        # self._task_name = data_config["task_name"]
        self._data_type = data_config["data_type"]

        # self._exp_name = data_config.get("exp_name", "")

        # Resolve the repository and data roots used by all storage formats.
        self._base_repo_path = data_config["repo_path"]
        self._base_data_path = data_config["base_data_path"]
        # self._base_task_path = data_config["task_path"]

        # Always normalize with training statistics, including validation/test data.
        self._dataset_path = f"{self._base_data_path}/datasets/{self._data_type}"
        self._stats_data_path = f"{self._base_data_path}/stats/{TRAIN_NAME}"  # by default, use train dataset scaling
        self._data_size_path = f"{self._base_data_path}/stats/{self._data_type}/data_size.npy"

        logger.info(f"base_data_path is {self._base_data_path}")
        # logger.info(f"base_task_path is {self._base_task_path}")
        logger.info(f"stats_data_path is {self._stats_data_path}")
        logger.info(f"dataset_path is {self._dataset_path}")
        logger.info(f"data_size_path is {self._data_size_path}")

        # HDF5 shards are addressed by a numeric format placeholder.
        self._dataset_h5_path = f"{self._dataset_path}" + "/data_{}.h5"

        logger.info(f"dataset_h5_path is {self._dataset_h5_path}")

        # TFRecord shards follow the same numeric naming convention.
        self._tfrecord_data_path = f"{self._dataset_path}" + "/data_{}.tfrecord"

        logger.info(f"tfrecord_data_path is {self._tfrecord_data_path}")

        # Concrete datasets must provide schemas before loading records.
        self._context_description: Optional[Dict[str, str]] = None
        self._feature_description: Optional[Dict[str, str]] = None

        self._validation()

        # Subclasses install their sample transformation pipeline here.
        self._transform: Optional[transforms.Compose] = None

        logger.info(f"=== Init {data_config['data_type']} data config done ===")

    def __len__(self) -> int:
        """Return the persisted number of samples in this dataset split."""
        return int(np.load(self._data_size_path))

    def _validation(self):
        """Ensure that dataset preparation recorded the split size."""
        if not os.path.exists(self._data_size_path):
            raise IOError(f"Data size file not found: {self._data_size_path}")

    def _init_transform(self) -> None:
        """Initialize a sample transform in a concrete dataset class."""
        return

    @abc.abstractmethod
    def get_head_inputs(self, batch_size) -> Dict:
        """Return representative inputs from a concrete dataset class."""
        raise NotImplementedError("Subclasses must implement get_head_inputs method")


class IndexedTrainDataset(BaseTrainDataset, Dataset):
    """Base class for random-access datasets addressed by sample index."""

    def get_head_inputs(self, batch_size) -> Dict:
        """Get head inputs for model inspection.

        Args:
            batch_size (int): Size of batch to generate
            gpu (bool): Whether to keep tensors on GPU

        Returns:
            Dict: Dictionary of input tensors
        """
        try:
            inputs, _ = self.__getitem__(np.arange(0, batch_size))

            return {key: data for key, data in inputs.items()}
        except Exception as e:
            logger.error(f"Failed to get head inputs: {e}")
            return {}


class BaseIterableDataset(BaseTrainDataset, IterableDataset):
    """Base class for iterable training datasets.

    Provides functionality for streaming data loading and batch generation.
    Subclasses implement specific data iteration logic.
    """

    def __init__(self, data_config: Dict, *args, **kwargs) -> None:
        """Initialize file counts and streaming shuffle configuration."""
        super().__init__(data_config, *args, **kwargs)
        # Each file represents one loadable dataset shard.
        self._num_of_files = len(os.listdir(self._dataset_path))

        # A missing queue size leaves the source ordering unchanged.
        self._shuffle_queue_size = data_config.get("shuffle_queue_size", None)

    def __iter__(self):
        """Return a sample iterator from a concrete streaming dataset."""
        raise NotImplementedError("Subclasses must implement __iter__ method")

    def get_head_inputs(self, batch_size) -> Dict:
        """Get head inputs for model inspection from iterable dataset.

        Args:
            batch_size (int): Size of batch to generate

        Returns:
            Dict: Dictionary of input tensors
        """
        try:
            res = {}

            for i in range(batch_size):
                inputs, _ = next(self.__iter__())
                inputs = {key: inputs[key].unsqueeze(0) for key in inputs}

                for key in inputs:
                    res[key] = torch.concat([res[key], inputs[key]], dim=0) if key in res else inputs[key]

            return {key: data for key, data in res.items()}

        except Exception as e:
            logger.error(f"Failed to get head inputs from iterable dataset: {e}")
            return {}


class MultiTFRecordDataset(BaseIterableDataset):
    """Dataset for loading multiple TFRecord files.

    Handles distributed data loading and shuffling of TFRecord format data.
    """

    def __init__(self, data_config: Dict, *args, **kwargs) -> None:
        """Initialize an uncompressed multi-file TFRecord dataset."""
        super().__init__(data_config, *args, **kwargs)
        # config
        # === file compression
        self._compression_type = None

    def _init_transform(self):
        """Initialize TFRecord sample transforms in a concrete subclass."""
        return

    def __iter__(self) -> (Dict, torch.Tensor):
        """Iterate over worker-partitioned TFRecord shards."""
        shift, num_workers = 0, 0

        worker_info = get_worker_info()
        if worker_info is not None:
            # Give each worker a deterministic seed and a disjoint shard offset.
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
            shift, num_workers = worker_info.id, worker_info.num_workers

        if num_workers > self._num_of_files:
            raise ValueError("the num of workers should be small or equal to num of files")

        if num_workers == 0:
            splits = {str(num): 1.0 for num in range(self._num_of_files)}
        else:
            splits = {str(num): 1.0 for num in range(self._num_of_files) if num % num_workers == shift}

        it = tfrecord.multi_tfrecord_loader(
            data_pattern=self._tfrecord_data_path,
            index_pattern=None,
            splits=splits,
            description=self._context_description,
            sequence_description=self._feature_description,
            compression_type=self._compression_type,
            infinite=False,
        )

        if self._shuffle_queue_size:
            it = tfrecord.shuffle_iterator(it, self._shuffle_queue_size)  # noqa

        it = map(self._transform, it)

        return it


class MultiHDF5Dataset(BaseIterableDataset):
    """Dataset for loading multiple HDF5 files.

    Handles distributed data loading and shuffling of HDF5 format data.
    """

    def _init_transform(self):
        """Initialize HDF5 sample transforms in a concrete subclass."""
        return

    def __iter__(self) -> (Dict, torch.Tensor):
        """Iterate over worker-partitioned HDF5 shards."""
        shift, num_workers = 0, 0

        worker_info = get_worker_info()
        if worker_info is not None:
            # Give each worker a deterministic seed and a disjoint shard offset.
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
            shift, num_workers = worker_info.id, worker_info.num_workers

        if num_workers > self._num_of_files:
            raise ValueError(
                f"the num of workers({num_workers}) should be small or equal to num of files({self._num_of_files})"
            )

        if num_workers == 0:
            splits = {str(num) for num in range(self._num_of_files)}
        else:
            splits = {str(num) for num in range(self._num_of_files) if num % num_workers == shift}

        # logger.info(f"worker {shift}/{num_workers}, currently deal with file {splits}")

        it = multi_hdf5_loader(
            data_pattern=self._dataset_h5_path,
            splits=splits,
            infinite=False,
            description=self._context_description,
            sequence_description=self._feature_description,
        )

        if self._shuffle_queue_size:
            it = shuffle_iterator(it, self._shuffle_queue_size)  # noqa

        it = map(self._transform, it)

        return it
