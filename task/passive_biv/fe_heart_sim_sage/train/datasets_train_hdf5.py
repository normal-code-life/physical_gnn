from typing import Dict

import numpy as np
from torchvision import transforms

from common.constant import TEST_NAME, TRAIN_NAME, VALIDATION_NAME
from pkg.train.datasets.base_datasets_train import MultiHDF5Dataset
from pkg.train.module.data_transform import ConvertToModelInputs, MaxMinNorm, ToTensor


class FEHeartSimSageTrainDataset(MultiHDF5Dataset):
    """Data loader for graph-formatted input-output data with common, fixed topology.

    This dataset loads and preprocesses training data for the FE Heart Sage model.
    It applies a series of transformations to prepare the data for training:
    1. Converts HDF5 data to PyTorch tensors
    2. Normalizes node coordinates, fiber/sheet orientations, shape coefficients, material parameters and pressures
    3. Clamps stress values to valid ranges
    4. Normalizes displacements and stresses
    5. Squeezes unnecessary dimensions from parameters
    6. Converts data into the format expected by the model

    Args:
        data_config (Dict): Configuration dictionary containing dataset parameters
    """

    def __init__(self, data_config: Dict) -> None:
        super().__init__(data_config)
        # node related features
        # === save data path
        self._node_coord_stats_path = f"{self._stats_data_path}/node_coord_stats.npz"
        self._node_laplace_coord_stats_path = f"{self._stats_data_path}/node_laplace_stats.npz"
        self._fiber_and_sheet_stats_path = f"{self._stats_data_path}/fiber_and_sheet_stats.npz"

        # global features
        # === save data path
        self._mat_param_stats_path = f"{self._stats_data_path}/mat_param_stats.npz"
        self._pressure_stats_path = f"{self._stats_data_path}/pressure_stats.npz"
        self._shape_coeff_stats_path = f"{self._stats_data_path}/shape_coeff_stats.npz"

        # label
        # === save data path
        self._displacement_stats_path = f"{self._stats_data_path}/displacement_stats.npz"
        self._stress_stats_path = f"{self._stats_data_path}/stress_stats.npz"

        # other
        self._normalize_val_objective = data_config["normalize_val_objective"]

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
            "time": "float",
        }

        self._labels = {"displacement"}

        self._init_transform()

    def __len__(self) -> int:
        return np.load(self._data_size_path).astype(np.int64).item()

    # init transform data
    def _init_transform(self):
        """Initialize the data transformation pipeline.

        Sets up a sequence of transforms to preprocess the raw data:
        - Convert HDF5 to tensors
        - Normalize various input features
        - Clamp stress values
        - Normalize displacements and stresses
        - Adjust data dimensions
        - Format for model input
        """
        transform_list = []

        hdf5_to_tensor_config = {
            "context_description": self._context_description,
            "feature_description": self._feature_description,
        }
        transform_list.append(ToTensor(hdf5_to_tensor_config))

        norm_config = {
            "node_coord": self._node_coord_stats_path,
            "fiber_and_sheet": self._fiber_and_sheet_stats_path,
            "shape_coeffs": self._shape_coeff_stats_path,
            "mat_param": self._mat_param_stats_path,
            "pressure": self._pressure_stats_path,
            # "time": self._time_stats_path,
        }

        transform_list.append(MaxMinNorm(norm_config, True, True))

        if self._data_type == TRAIN_NAME or (
            self._data_type in (VALIDATION_NAME, TEST_NAME) and self._normalize_val_objective
        ):
            norm_config_disp = {
                "displacement": self._displacement_stats_path,
            }
            transform_list.append(MaxMinNorm(norm_config_disp))

        # convert data dim
        # convert_data_dim_config = {"mat_param": -1, "pressure": -1, "shape_coeffs": -1, "time": -1}
        # transform_list.append(SqueezeDataDim(convert_data_dim_config))

        # convert to model inputs
        convert_model_input_config = {"labels": self._labels}

        transform_list.append(ConvertToModelInputs(convert_model_input_config, True))

        self._transform = transforms.Compose(transform_list)

    def get_head_inputs(self, batch_size) -> Dict:
        """Get a batch of inputs for model visualization/debugging.

        Creates a batch of inputs by sampling from the dataset iterator.
        Also adds randomly selected nodes for the fe_heart_sage_v4 model version.

        Args:
            batch_size (int): Number of samples to include in the batch

        Returns:
            Dict: Batch of model inputs with shape [batch_size, ...]
        """
        res = super().get_head_inputs(batch_size)

        _, node_num, _ = res["edges_indices"].shape

        res["selected_node"] = np.arange(node_num)
        res["select_edge_num"] = 12

        return res
