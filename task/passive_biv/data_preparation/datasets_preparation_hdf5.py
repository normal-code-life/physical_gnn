from typing import Dict, List, Optional

import h5py
import numpy as np
from numba.typed import List as Numba_List

from common.constant import DARWIN, TRAIN_NAME, VALIDATION_NAME
from pkg.data.base_datasets_preparation import AbstractDataPreparationDataset
from pkg.utils.data_utils.edge_generation import generate_distance_based_edges_nb, generate_distance_based_edges_ny
from pkg.utils.data_utils.stats import stats_analysis
from task.passive_biv.data_preparation import logger


class PassiveBiVPreparationDataset(AbstractDataPreparationDataset):
    """Dataset preparation class for passive bi-ventricular heart model.

    Handles loading, preprocessing and saving heart simulation data into HDF5 format.
    Includes functionality for:
    - Loading node coordinates, material parameters, pressures etc.
    - Generating distance-based edges between nodes
    - Computing dataset statistics
    - Down sampling nodes if needed
    - Saving processed data into chunked HDF5 files

    Args:
        data_config (Dict): Configuration dictionary containing:
    """

    def __init__(self, data_config: Dict) -> None:
        super().__init__(data_config)

        # node related features
        # === read data path
        self._inputs_data_path = f"{self._base_data_path}/record_inputs"

        # === save data path
        self._node_coord_stats_path = f"{self._stats_data_path}/node_coord_stats.npz"
        self._node_laplace_coord_stats_path = f"{self._stats_data_path}/node_laplace_stats.npz"
        self._fiber_and_sheet_stats_path = f"{self._stats_data_path}/fiber_and_sheet_stats.npz"

        logger.info(f"inputs_data_path is {self._inputs_data_path}")
        logger.info(f"node_coord_stats_path is {self._node_coord_stats_path}")
        logger.info(f"node_laplace_coord_stats_path is {self._node_laplace_coord_stats_path}")
        logger.info(f"fiber_and_sheet_stats_path is {self._fiber_and_sheet_stats_path}")

        # global features
        # === read data path
        self._global_feature_data_path = f"{self._base_data_path}/record_global_feature.csv"
        self._shape_data_path = f"{self._base_data_path}/record_shape.csv"

        logger.info(f"global_feature_data_path is {self._global_feature_data_path}")
        logger.info(f"shape_data_path is {self._shape_data_path}")

        # === save data path
        self._mat_param_stats_path = f"{self._stats_data_path}/mat_param_stats.npz"
        self._pressure_stats_path = f"{self._stats_data_path}/pressure_stats.npz"
        self._time_stats_path = f"{self._stats_data_path}/time_stats.npz"
        self._shape_coeff_stats_path = f"{self._stats_data_path}/shape_coeff_stats.npz"

        logger.info(f"mat_param_stats_path is {self._mat_param_stats_path}")
        logger.info(f"pressure_stats_path is {self._pressure_stats_path}")
        logger.info(f"time_stats_path is {self._time_stats_path}")
        logger.info(f"shape_coeff_stats_path is {self._shape_coeff_stats_path}")

        # label
        # === read data path
        self._outputs_data_path = f"{self._base_data_path}/record_results"

        # === save data path
        self._displacement_stats_path = f"{self._stats_data_path}/displacement_stats.npz"
        self._stress_stats_path = f"{self._stats_data_path}/stress_stats.npz"

        logger.info(f"outputs_data_path is {self._outputs_data_path}")
        logger.info(f"displacement_stats_path is {self._displacement_stats_path}")
        logger.info(f"stress_stats_path is {self._stress_stats_path}")

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

        self._labels = {"displacement", "stress"}

        # sample indices
        self._sample_indices = data_config["sample_indices"]

        # other parameter
        # === param sample size for each files
        self._chunk_file_size = data_config["chunk_file_size"]

        # === param random select edges based on node relative distance
        self._sections = data_config["sections"]
        self._nodes_per_sections = data_config["nodes_per_sections"]
        self._train_down_sampling_node: Optional[float] = data_config.get("train_down_sampling_node", None)
        self._val_down_sampling_node: Optional[float] = data_config.get("val_down_sampling_node", None)

        logger.info(f"====== finish {self.__class__.__name__} {self._data_type} data config ======")

    def _data_generation(self):
        """Generate processed dataset and save to HDF5 files.

        Loads raw data, processes it and saves into chunked HDF5 files containing:
        - Node coordinates and features
        - Edge indices
        - Material parameters
        - Pressures
        - Shape coefficients
        - Ground truth displacements and stresses
        """
        # read global features
        data_global_feature = np.loadtxt(self._global_feature_data_path, delimiter=",")
        data_shape_coeff = np.loadtxt(self._shape_data_path, delimiter=",")

        sample_indices: List[np.ndarray] = np.array_split(
            self._sample_indices, len(self._sample_indices) // self._chunk_file_size
        )

        for i, indices in enumerate(sample_indices):
            datasets = []

            for idx in indices:
                # read sample inputs
                read_file_name = f"/ct_case_{idx + 1:04d}.csv"  # e.g. ct_case_0005

                record_inputs = np.loadtxt(self._inputs_data_path + read_file_name, delimiter=",", dtype=np.float32)

                record_outputs = np.loadtxt(self._outputs_data_path + read_file_name, delimiter=",", dtype=np.float32)

                points = record_inputs.shape[0]

                if (self._data_type == VALIDATION_NAME and self._val_down_sampling_node) or (
                    self._data_type == TRAIN_NAME and self._train_down_sampling_node
                ):
                    record_inputs, record_outputs = self._down_sampling_node(record_inputs, record_outputs)

                edge: np.ndarray = self._generate_distance_based_edges(record_inputs[:, 0:3])

                datasets.append(
                    {
                        "index": np.array([np.int32(idx + 1)]),
                        "points": np.array([np.int32(points)]),
                        "node_coord": record_inputs[:, 0:3],
                        "laplace_coord": record_inputs[:, 3:11],
                        "fiber_and_sheet": record_inputs[:, 11:17],
                        "edges_indices": edge[0].astype(np.int32),
                        "mat_param": data_global_feature[:, 1:7][idx],
                        "pressure": data_global_feature[:, 7:9][idx],
                        "time": data_global_feature[:, 9:10][idx],
                        "shape_coeffs": data_shape_coeff[:, 1:59][idx],
                        "displacement": record_outputs[:, 0:3],
                        "stress": record_outputs[:, 3:4],
                    }
                )

            with h5py.File(self._dataset_h5_path.format(i), "w") as f:
                for idx, sample_dict in enumerate(datasets):
                    group: h5py.Group = f.create_group(f"idx_{idx}")
                    for key, value in sample_dict.items():
                        group.create_dataset(key, data=value)

            logger.info(f"data_generation {i}: {indices} done")

    def _down_sampling_node(self, record_inputs: np.ndarray, record_outputs: np.ndarray) -> (np.ndarray, np.ndarray):
        """Randomly downsample nodes from input and output data.

        Args:
            record_inputs (np.ndarray): Input features for all nodes
            record_outputs (np.ndarray): Output values for all nodes

        Returns:
            tuple: Downsampled input and output arrays

        Raises:
            ValueError: If requested number of nodes exceeds available nodes
        """
        num_nodes, record_inputs_dim = record_inputs.shape

        _, record_outputs_dim = record_outputs.shape

        num_down_sample_node = (
            self._train_down_sampling_node if self._data_type == TRAIN_NAME else self._val_down_sampling_node
        )

        if num_down_sample_node > num_nodes:
            raise ValueError("num_down_sample_node error, please carefully choice the node number")

        select_nodes = np.random.choice(num_nodes, size=num_down_sample_node, replace=False)

        return record_inputs[select_nodes, :], record_outputs[select_nodes, :]

    def _generate_distance_based_edges(self, node_coords) -> np.ndarray:
        """Generate edges between nodes based on distances.

        Uses platform-specific implementation (Darwin vs others) to generate
        edges connecting nodes within specified distance thresholds.

        Args:
            node_coords: Node coordinate array

        Returns:
            np.ndarray: Edge indices array
        """
        if self._platform == DARWIN:
            return generate_distance_based_edges_ny(
                node_coords[np.newaxis, :, :], [0], self._sections, self._nodes_per_sections
            )

        sections_nb = Numba_List()
        [sections_nb.append(x) for x in self._sections]

        nodes_per_section_nb = Numba_List()
        [nodes_per_section_nb.append(x) for x in self._nodes_per_sections]

        # need to expand the axis and align with the other method
        return generate_distance_based_edges_nb(node_coords, sections_nb, nodes_per_section_nb)[np.newaxis, :].astype(
            np.int32
        )

    def _data_stats_generation(self) -> None:
        """Compute statistics for all data components.

        Calculates and saves statistics for:
        - Node features (coordinates, Laplace coordinates, fiber orientations)
        - Global features (material parameters, pressures, shape coefficients)
        - Labels (displacements, stresses)
        """
        self._data_node_stats()

        self._data_global_feature_stats()

        self._data_label_stats()

        self._data_stats_total_size()

    def _data_node_stats(self, write_to_path: bool = True) -> None:
        """Compute statistics for node-level features.

        Args:
            write_to_path (bool): Whether to save stats to files
        """
        # fmt: off
        node_coord_set: Optional[np.ndarray] = None
        laplace_coord_set: Optional[np.ndarray] = None
        fiber_and_sheet_set: Optional[np.ndarray] = None

        for idx in range(len(self._sample_indices)):
            read_file_name = f"/ct_case_{idx + 1:04d}.csv"  # e.g. ct_case_0005
            record_inputs = np.loadtxt(self._inputs_data_path + read_file_name, delimiter=",")

            node_coord = record_inputs[:, 0:3]
            node_coord_set = (
                node_coord if node_coord_set is None else np.concatenate([node_coord_set, node_coord], axis=0)
            )

            laplace_coord = record_inputs[:, 3:11]
            laplace_coord_set = (
                laplace_coord if laplace_coord_set is None else np.concatenate([laplace_coord_set, laplace_coord], axis=0)  # noqa
            )

            fiber_and_sheet = record_inputs[:, 11:17]
            fiber_and_sheet_set = (
                fiber_and_sheet if fiber_and_sheet_set is None else np.concatenate([fiber_and_sheet_set, fiber_and_sheet], axis=0)  # noqa
            )

        stats_analysis("node_coord", node_coord_set, 0, self._node_coord_stats_path, logger, write_to_path)  # noqa
        stats_analysis("laplace_coord", laplace_coord_set, 0, self._node_laplace_coord_stats_path, logger, write_to_path)  # noqa
        stats_analysis("fiber_and_sheet", fiber_and_sheet_set, 0, self._fiber_and_sheet_stats_path, logger, write_to_path)  # noqa

        # fmt: on

    def _data_global_feature_stats(self, write_to_path: bool = True) -> None:
        """Compute statistics for global features.

        Args:
            write_to_path (bool): Whether to save stats to files
        """
        # fmt: off
        data_global_feature = np.loadtxt(self._global_feature_data_path, delimiter=",")
        data_shape_coeff = np.loadtxt(self._shape_data_path, delimiter=",")

        stats_analysis("mat_param", data_global_feature[self._sample_indices, 1:7], 0, self._mat_param_stats_path, logger, write_to_path)  # noqa
        stats_analysis("pressure", data_global_feature[self._sample_indices, 7:9], 0, self._pressure_stats_path, logger, write_to_path)  # noqa
        stats_analysis("time", data_global_feature[self._sample_indices, 9:10], 0, self._time_stats_path, logger, write_to_path)  # noqa
        stats_analysis("shape_coeffs", data_shape_coeff[self._sample_indices, 1:], 0, self._shape_coeff_stats_path, logger, write_to_path)  # noqa

        # fmt: on

    def _data_label_stats(self, write_to_path: bool = True) -> None:
        """Compute statistics for output labels.

        Args:
            write_to_path (bool): Whether to save stats to files
        """
        # fmt: off
        displacement_set: Optional[np.ndarray] = None
        stress_set: Optional[np.ndarray] = None

        for idx in range(len(self._sample_indices)):
            read_file_name = f"/ct_case_{idx + 1:04d}.csv"  # e.g. ct_case_0005
            record_output = np.loadtxt(self._outputs_data_path + read_file_name, delimiter=",")

            displacement = record_output[:, 0: 3]
            displacement_set = (
                displacement if displacement_set is None else np.concatenate([displacement_set, displacement], axis=0)
            )

            stress = record_output[:, 3: 4]
            stress_set = (
                stress if stress_set is None else np.concatenate([stress_set, stress], axis=0)
            )

        stats_analysis("displacement", displacement_set, 0, self._displacement_stats_path, logger, write_to_path)  # noqa
        stats_analysis("stress", stress_set, 0, self._stress_stats_path, logger, write_to_path)

        # fmt: on

    def _data_stats_total_size(self) -> None:
        """Save total number of samples to file."""
        np.save(self._data_size_path, self._sample_indices.shape[0])
