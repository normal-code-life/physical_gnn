"""Prepare single passive BiV cases and export quantitative evaluation results."""

import platform
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchmetrics
from numba.typed import List as Numba_List
from torch import Tensor, nn
from torchvision import transforms

from common.constant import DARWIN, MAX_VAL, MIN_VAL
from pkg.train.datasets.base_datasets_train import MultiHDF5Dataset
from pkg.train.module.data_transform import ConvertToModelInputs, MaxMinNorm, ToTensor, UnSqueezeDataDim
from pkg.train.module.loss import EuclideanDistanceMSE
from pkg.utils.data_utils.edge_generation import generate_distance_based_edges_nb, generate_distance_based_edges_ny
from pkg.utils.other_utils.logs import init_logger

logger = init_logger("SINGLE_CASE_EVAL")


class FEHeartSimSageEvaluation(MultiHDF5Dataset):
    """Evaluation class for FE Heart SIM SAGE model that handles single graph cases.

    This class extends FEHeartSageDataset to provide evaluation functionality for individual test cases.
    It handles data loading, preprocessing, model inference and result saving for single graph evaluations.
    The class supports configurable data transforms and model loading from checkpoints.
    """

    def __init__(self, data_config: Dict) -> None:
        """Initialize the evaluation class.

        Args:
            data_config (Dict): Configuration dictionary containing data parameters
        """
        super().__init__(data_config)
        self._platform = platform.system()
        # node related features
        # === read data path
        self._inputs_data_path = f"{self._base_data_path}/record_inputs"
        logger.info(f"inputs_data_path is {self._inputs_data_path}")

        # === save data path
        self._node_coord_stats_path = f"{self._stats_data_path}/node_coord_stats.npz"
        self._node_laplace_coord_stats_path = f"{self._stats_data_path}/node_laplace_stats.npz"
        self._fiber_and_sheet_stats_path = f"{self._stats_data_path}/fiber_and_sheet_stats.npz"

        # global features
        # === read data path
        self._global_feature_data_path = f"{self._base_data_path}/record_global_feature.csv"
        self._shape_data_path = f"{self._base_data_path}/record_shape.csv"
        logger.info(f"global_feature_data_path is {self._global_feature_data_path}")
        logger.info(f"shape_data_path is {self._shape_data_path}")

        # === save data path
        self._mat_param_stats_path = f"{self._stats_data_path}/mat_param_stats.npz"
        self._pressure_stats_path = f"{self._stats_data_path}/pressure_stats.npz"
        self._shape_coeff_stats_path = f"{self._stats_data_path}/shape_coeff_stats.npz"

        # label
        # === read data path
        self._outputs_data_path = f"{self._base_data_path}/record_results"
        logger.info(f"outputs_data_path is {self._outputs_data_path}")

        # === save data path
        self._displacement_stats_path = f"{self._stats_data_path}/displacement_stats.npz"
        self._stress_stats_path = f"{self._stats_data_path}/stress_stats.npz"

        # other
        self._down_sample_node = data_config.get("down_sample_node")

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

        # data preparation param
        # === test case number
        # self.idx = idx
        self.device = data_config["device"]

        # === param random select edges based on node relative distance
        self.sections = data_config["sections"]
        self.nodes_per_sections = data_config["nodes_per_sections"]

        self.output_path = data_config["output_path"]

        self._init_transform()

    def generate_single_inputs(self, idx: int):
        """Evaluate a single graph case and save results.

        Generates data, applies transforms, runs model inference and saves output to CSV.
        """
        data = self._data_generation(idx)

        inputs, labels = self._transform(data)

        return inputs, labels

    def _data_generation(self, idx: int) -> (Dict[str, np.ndarray], Dict[str, np.ndarray]):
        """Generate input and output data for a single test case.

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: Tuple containing:
                - context_example: Dictionary with index and points information
                - feature_example: Dictionary with node features and labels
        """
        # read global features
        data_global_feature = np.loadtxt(self._global_feature_data_path, delimiter=",")
        data_shape_coeff = np.loadtxt(self._shape_data_path, delimiter=",")

        read_file_name = f"/ct_case_{idx + 1:04d}.csv"  # e.g. ct_case_0005
        record_inputs = np.loadtxt(self._inputs_data_path + read_file_name, delimiter=",", dtype=np.float32)
        record_outputs = np.loadtxt(self._outputs_data_path + read_file_name, delimiter=",", dtype=np.float32)

        points = record_inputs.shape[0]

        if self._down_sample_node is not None:
            record_inputs, record_outputs = self._down_sampling_node(record_inputs, record_outputs)

        edge: np.ndarray = self._generate_distance_based_edges(record_inputs[:, 0:3])

        context_example = {
            "index": np.array([np.int32(idx)]),
            "points": np.array([np.int32(points)]),
        }

        feature_example = {
            "node_coord": record_inputs[:, 0:3],
            "laplace_coord": record_inputs[:, 3:11],
            "fiber_and_sheet": record_inputs[:, 11:17],
            "edges_indices": edge[0].astype(np.int64),
            "mat_param": data_global_feature[:, 1:7][idx],
            "pressure": data_global_feature[:, 7:9][idx],
            "time": data_global_feature[:, 9:10][idx],
            "shape_coeffs": data_shape_coeff[:, 1:59][idx],
            "displacement": record_outputs[:, 0:3],
            "stress": record_outputs[:, 3:4],
        }

        return context_example, feature_example

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

        num_down_sample_node = self._down_sample_node

        if num_down_sample_node > num_nodes:
            raise ValueError("num_down_sample_node error, please carefully choice the node number")

        select_nodes = np.random.choice(num_nodes, size=num_down_sample_node, replace=False)

        return record_inputs[select_nodes, :], record_outputs[select_nodes, :]

    def _generate_distance_based_edges(self, node_coords) -> np.ndarray:
        """Generate edges based on node distances.

        Args:
            node_coords: Node coordinates array

        Returns:
            np.ndarray: Generated edges array
        """
        if self._platform == DARWIN:
            return generate_distance_based_edges_ny(
                node_coords[np.newaxis, :, :], [0], self.sections, self.nodes_per_sections
            )

        sections = self.sections
        nodes_per_sections = self.nodes_per_sections

        sections_nb = Numba_List()
        [sections_nb.append(x) for x in sections]

        nodes_per_section_nb = Numba_List()
        [nodes_per_section_nb.append(x) for x in nodes_per_sections]

        # need to expand the axis and align with the other method
        return generate_distance_based_edges_nb(node_coords, sections_nb, nodes_per_section_nb)[np.newaxis, :].astype(
            np.int32
        )

    def _init_transform(self):
        """Initialize data transformation pipeline.

        Returns:
            transforms.Compose: Composed transformation pipeline
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
        }

        transform_list.append(MaxMinNorm(norm_config, True, True))

        # norm_config = {
        #     "displacement": self._displacement_stats_path,
        # }
        # transform_list.append(MaxMinNorm(norm_config))

        unsqueeze_data_dim_config = {
            "node_coord": 0,
            "laplace_coord": 0,
            "fiber_and_sheet": 0,
            "edges_indices": 0,
            "time": 0,
            "displacement": 0,
            "stress": 0,
            "mat_param": 0,
            "pressure": 0,
            "shape_coeffs": 0,
        }
        transform_list.append(UnSqueezeDataDim(unsqueeze_data_dim_config))

        convert_model_input_config = {"labels": self._labels}

        transform_list.append(ConvertToModelInputs(convert_model_input_config, True))

        self._transform = transforms.Compose(transform_list)

        return self._transform

    @staticmethod
    def total_params_count(model: nn.Module) -> None:
        """Print the model structure and its total and trainable parameter counts."""
        logger.info(f"print model arch: {model}")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n{'=' * 50}")
        print("Model Architecture:")
        print(model)
        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"{'=' * 50}\n")

    def convert_outputs(self, outputs: torch.tensor) -> torch.tensor:
        """Convert normalized displacement predictions back to physical units."""
        stats = np.load(self._displacement_stats_path)

        max_val = torch.tensor(stats[MAX_VAL], device=self.device)
        min_val = torch.tensor(stats[MIN_VAL], device=self.device)

        return outputs["displacement"] * (max_val - min_val) + min_val

    def calc_l1_loss(self, outputs, labels) -> torch.tensor:
        """Return mean absolute error over all displacement components."""
        return torchmetrics.functional.mean_absolute_error(outputs, labels)

    def calc_l2_loss(self, outputs, labels) -> torch.tensor:
        """Return mean Euclidean displacement error."""
        l2_loss = EuclideanDistanceMSE()
        return l2_loss(outputs, labels)

    def calc_l2_loss_perc(self, outputs, labels) -> torch.tensor:
        """Return mean vector-error magnitude relative to target magnitude."""
        labels_norm = torch.sqrt((torch.sum(labels**2, dim=-1)))
        gap_norm = torch.sqrt((torch.sum((outputs - labels) ** 2, dim=-1)))
        return torch.mean(gap_norm / labels_norm)

    def calc_l2_loss_perc2(self, outputs, labels) -> torch.tensor:
        """Return mean relative error between predicted and target magnitudes."""
        labels_norm = torch.sqrt((torch.sum(labels**2, dim=-1)))
        pred_norm = torch.sqrt((torch.sum(outputs**2, dim=-1)))
        return torch.mean(torch.abs(pred_norm - labels_norm) / labels_norm)

    def sample_l1_loss(self, outputs, labels) -> torch.tensor:
        """Return signed component-wise errors for every sample node."""
        l1_loss = outputs - labels
        return l1_loss

    def sample_l2_loss(self, outputs, labels) -> torch.tensor:
        """Return Euclidean displacement error for every sample node."""
        l2_loss = torch.sqrt((torch.sum((outputs - labels) ** 2, dim=-1, keepdim=True)))
        return l2_loss

    def save_single_outputs(self, outputs: torch.tensor, idx: int) -> torch.tensor:
        """Save one case's displacement predictions as a CSV file."""
        df = pd.DataFrame(outputs.squeeze(0).numpy())
        df.to_csv(f"{self.output_path}/output_{idx + 1:04d}.csv", index=False)

    def save_numpy_outputs(self, name, outputs) -> torch.tensor:
        """Save a named aggregate result object in NumPy format."""
        np.save(f"{self.output_path}/{name}.npy", outputs, allow_pickle=True)


class ConvertToModelInputsWithSelectedNode(ConvertToModelInputs):
    """Convert inputs with selected nodes for model processing."""

    def __init__(self, config: Dict, multi_obj: bool = False) -> None:
        """Initialize the converter.

        Args:
            config (Dict): Configuration dictionary
            multi_obj (bool, optional): Whether using multiple objectives. Defaults to False.
        """
        super().__init__(config, multi_obj)

    def __call__(
        self, sample: Tuple[Dict[str, Tensor], Dict[str, Tensor]]
    ) -> Tuple[Dict[str, Tensor], Union[Tensor, Dict[str, Tensor]]]:
        """Convert input sample to model format with selected nodes.

        Args:
            sample: Tuple of input and label dictionaries

        Returns:
            Tuple containing processed inputs and labels
        """
        inputs, labels = super().__call__(sample)

        batch_size, node_num, _ = inputs["edges_indices"].shape

        selected_node = torch.arange(node_num, device="cpu").unsqueeze(0).expand(batch_size, -1)

        inputs["selected_node"] = selected_node
        inputs["select_edge_num"] = torch.tensor(100, dtype=torch.int64)

        return inputs, labels
