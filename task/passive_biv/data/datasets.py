from typing import Dict, Optional

from pkg.train.datasets.base_datasets import BaseAbstractDataset
from task.passive_biv.data import logger


class FEHeartSimSageDataset(BaseAbstractDataset):
    """FE Heart Sage Dataset main class which including our basic attributes.

    This class is responsible for loading and processing data for a specific task,
    organized in a predefined directory structure. It supports reading from and saving
    to local paths, including data in tfrecord and npz formats. It also sets up the
    necessary paths for various data features and statistics.

    Parameters:
    ----------
    data_config : Dict
        A dictionary containing configuration information for the data. Expected keys include:
            - 'task_data_path': Base path for task-related data.
            - 'task_path': Path for task-specific files.
            - 'exp_name': (Optional) Name of the experiment.
            - 'default_padding_value': (Optional) Default padding value for data.

    data_type : str
        Type of data to be processed (e.g., 'train', 'test', 'validate').
    """

    def __init__(self, data_config: Dict, data_type: str, process: Optional[str] = None) -> None:
        super().__init__(data_config, data_type, process)

        # node related features
        # === read data path
        self._inputs_data_path = f"{self._base_data_path}/record_inputs"

        # === save data path
        self._node_coord_stats_path = f"{self._stats_data_path}/node_coord_stats.npz"
        self._node_laplace_coord_stats_path = f"{self._stats_data_path}/node_laplace_stats.npz"
        self._fiber_and_sheet_stats_path = f"{self._stats_data_path}/fiber_and_sheet_stats.npz"

        logger.info(f"inputs_data_path is {self._inputs_data_path}")

        # global features
        # === read data path
        self._global_feature_data_path = f"{self._base_data_path}/record_global_feature.csv"
        self._shape_data_path = f"{self._base_data_path}/record_shape.csv"

        # === save data path
        self._mat_param_stats_path = f"{self._stats_data_path}/mat_param_stats.npz"
        self._pressure_stats_path = f"{self._stats_data_path}/pressure_stats.npz"
        self._shape_coeff_stats_path = f"{self._stats_data_path}/shape_coeff_stats.npz"

        logger.info(f"global_feature_data_path is {self._global_feature_data_path}")
        logger.info(f"shape_data_path is {self._shape_data_path}")

        # label
        # === read data path
        self._outputs_data_path = f"{self._base_data_path}/record_results"

        # === save data path
        self._displacement_stats_path = f"{self._stats_data_path}/displacement_stats.npz"
        self._stress_stats_path = f"{self._stats_data_path}/stress_stats.npz"

        logger.info(f"outputs_data_path is {self._outputs_data_path}")

        # features
        self._context_description = {
            "index": "int",
            "points": "int",
        }

        self._feature_description = {
            "node_coord": "float",
            "laplace_coord": "float",
            "fiber_and_sheet": "float",
            "edges_indices": "int",
            "shape_coeffs": "float",
            "mat_param": "float",
            "pressure": "float",
            "displacement": "float",
            "stress": "float",
        }

        self._labels = {"displacement", "stress"}
