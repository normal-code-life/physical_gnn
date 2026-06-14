from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from common.constant import MAX_VAL, MIN_VAL, TRAIN_NAME
from pkg.train.layer.pooling_layer import MeanAggregator, SUMAggregator  # noqa
from pkg.train.model.base_model import BaseModule
from pkg.train.model.transformer import MultiHeadAttention
from pkg.train.trainer.base_trainer import BaseTrainer
from pkg.utils.other_utils.logs import init_logger
from task.passive_biv.fe_heart_sim_sage.train.datasets_train_hdf5 import FEHeartSimSageTrainDataset
from task.passive_biv.utils.module.mlp_layer_ln import MLPLayer

logger = init_logger("FE_PASSIVE_BIV_HEART_SAGE")

torch.manual_seed(753)
torch.set_printoptions(precision=8)


class FEHeartSimSageTrainer(BaseTrainer):
    dataset_class = FEHeartSimSageTrainDataset

    def __init__(self) -> None:
        super().__init__()

        # global tune hyper param
        self._select_node_num = self.task_data["select_node_num"]
        self._select_edge_num = self.task_data["select_edge_num"]

        # normalize val dataset
        self._normalize_val_objective = self.task_data["normalize_val_objective"]
        if not self._normalize_val_objective:
            base_data_path = self.task_data["base_data_path"]
            stats_data_path = f"{base_data_path}/stats/{TRAIN_NAME}"
            displacement_stats_path = f"{stats_data_path}/displacement_stats.npz"
            self._displacement_stats = np.load(displacement_stats_path)
            self._displacement_stats_min = torch.tensor(self._displacement_stats[MIN_VAL])
            self._displacement_stats_max = torch.tensor(self._displacement_stats[MAX_VAL])
            if self.task_train["device"] == "cuda":
                self._displacement_stats_min = torch.tensor(self._displacement_stats[MIN_VAL], device="cuda")
                self._displacement_stats_max = torch.tensor(self._displacement_stats[MAX_VAL], device="cuda")

    def create_model(self) -> None:
        self.model = FEHeartSimSAGEModel(self.task_train)

    def validation_step_check(self, epoch: int, is_last_epoch: bool) -> bool:
        return True

    def post_transform_data(
        self, data: (Union[Dict[str, Tensor], Tensor], Union[Dict[str, Tensor], Tensor])
    ) -> (Union[Dict[str, Tensor], Tensor], Union[Dict[str, Tensor], Tensor], Dict[str, Any]):
        inputs, labels, args = super().post_transform_data(data)

        _, node_num, _ = inputs["edges_indices"].shape
        selected_node = np.random.randint(0, node_num, size=self._select_node_num, dtype=np.int64)

        for label_name in self.labels:
            labels[label_name] = labels[label_name][:, selected_node, :]

        args.update(
            {
                "select_edge_num": self._select_edge_num,
                "selected_node": selected_node,
            }
        )

        return inputs, labels, args

    def post_transform_val_data(
        self, data: (Union[Dict[str, Tensor], Tensor], Union[Dict[str, Tensor], Tensor])
    ) -> (Union[Dict[str, Tensor], Tensor], Union[Dict[str, Tensor], Tensor], Dict[str, Any]):
        inputs, labels, args = super().post_transform_val_data(data)

        _, node_num, _ = inputs["edges_indices"].shape
        selected_node = np.arange(node_num)

        args.update(
            {
                "select_edge_num": self._select_edge_num,
                "selected_node": selected_node,
            }
        )

        return inputs, labels, args

    def compute_validation_loss(self, predictions: Dict[str, Tensor], labels: Dict[str, Tensor]) -> Dict[str, Tensor]:
        losses = dict()

        for label_name in self.labels:
            prediction = predictions[label_name]
            label = labels[label_name]

            if label_name == "displacement" and not self._normalize_val_objective:
                prediction = (
                    prediction * (self._displacement_stats_max - self._displacement_stats_min)
                    + self._displacement_stats_min
                )
            losses[label_name] = self.loss(prediction, label)

        return losses

    def compute_metrics(
        self, metrics_func: callable, predictions: Dict[str, Tensor], labels: Dict[str, Tensor]
    ) -> Union[Dict[str, Tensor], Tensor]:
        metrics = dict()

        for label_name in self.labels:
            prediction = predictions[label_name]
            label = labels[label_name]

            if label_name == "displacement" and not self._normalize_val_objective:
                prediction = (
                    prediction * (self._displacement_stats_max - self._displacement_stats_min)
                    + self._displacement_stats_min
                )

            metrics[label_name] = metrics_func(prediction, label)

        return metrics


class FEHeartSimSAGEModel(BaseModule):
    def __init__(self, config: Dict, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

        # mlp layer config
        self._input_layer_config = config["input_layer"]
        self._edge_mlp_layer_config = config["edge_mlp_layer"]
        self._edge_laplace_mlp_layer_config = config["edge_laplace_mlp_layer"]
        # self._theta_input_mlp_layer_config = config["theta_input_mlp_layer"]
        self._decoder_layer_config = config["decoder_layer"]

        # message config
        self._message_passing_layer_config = config["message_passing_layer"]

        self._device = config["device"]

        self._init_graph()

    def get_config(self) -> Dict:
        base_config = super().get_config()

        mlp_config = {
            "node_input_mlp_layer": self.node_input_mlp_layer,
            # "theta_input_mlp_layer": self._theta_input_mlp_layer_config,
            "message_config": self.message_layer_config,
            "decoder_layer_config": self._decoder_layer_config,
            "device": self._device,
        }

        return {**base_config, **mlp_config}

    def _init_graph(self):
        # Input layer
        self._input_layer: nn.ModuleList = nn.ModuleList()
        for layer_name, layer_config in self._input_layer_config.items():
            self._input_layer.append(MLPLayer(layer_config, prefix_name=layer_name))

        self._edge_mlp_layer = MLPLayer(self._edge_mlp_layer_config, prefix_name="edge_input")

        self._edge_laplace_mlp_layer = MLPLayer(self._edge_laplace_mlp_layer_config, prefix_name="edge_laplace_input")

        if self._message_passing_layer_config["arch"] == "attention":
            # self._message_update_layer = nn.TransformerEncoderLayer(
            #     d_model=self._message_passing_layer_config["message_update_layer"].get("d_model", 128),
            #     nhead=self._message_passing_layer_config["message_update_layer"].get("nhead", 4),
            #     dim_feedforward=self._message_passing_layer_config["message_update_layer"].get("dim_feedforward", 512)
            #     dropout=self._message_passing_layer_config["message_update_layer"].get("dropout", 0.1),
            #     device=self._device,
            #     batch_first=True,
            # )
            self._message_update_layer = MultiHeadAttention(
                self._message_passing_layer_config["message_update_layer"], prefix_name="message_att"
            )
            self._message_update_layer_mlp = MLPLayer(
                self._message_passing_layer_config["message_update_layer_mlp"], prefix_name="message"
            )
        elif self._message_passing_layer_config["arch"] == "mlp":
            self._message_update_layer = MLPLayer(
                self._message_passing_layer_config["message_update_layer"], prefix_name="message"
            )
        else:
            raise ValueError(
                f"please define the arch properly, current is {self._message_passing_layer_config['arch']}"
            )

        # aggregator pooling
        agg_method = self._message_passing_layer_config["agg_method"]
        self._message_agg_pooling = globals()[agg_method](self._message_passing_layer_config["agg_layer"])

        # theta mlp
        # self._theta_encode_mlp_layer = MLPLayer(self._theta_input_mlp_layer_config, prefix_name="theta_encode")

        # decoder MLPs
        decoder_layer_config = self._decoder_layer_config
        self._decoder_layer = nn.ModuleList(
            [
                MLPLayer(decoder_layer_config, prefix_name=f"decode_{i}")
                for i in range(decoder_layer_config["output_dim"])
            ]
        )

    @staticmethod
    def _random_select_edge(indices: Tensor, device: str, selected_edge_num: int) -> Tensor:
        batch_size, node_num, seq_num = indices.shape

        select_batch = torch.arange(batch_size, device=indices.device)

        select_indices = torch.randint(
            0,
            seq_num,
            (batch_size, node_num, selected_edge_num),
            dtype=torch.int64,
            device=indices.device,
        )

        selected_node = torch.arange(node_num, device=indices.device)

        return indices[
            select_batch[:, None, None], selected_node[None, :, None], select_indices
        ]  # shape: (batch_size, selected_node_num, seq)

    @staticmethod
    def _generate_edge_emb(node_emb: Tensor, input_edge_indices: Tensor) -> Tensor:
        emb_dim: int = node_emb.shape[-1]  # feature for each of the node
        seq: int = input_edge_indices.shape[-1]  # neighbours seq for each of the center node

        # === expand node feature to match indices shape
        # shape: (batch_size, node_num, emb) =>
        # (batch_size, node_num, 1, emb) =>
        # (batch_size, node_num, seq, emb)
        node_emb_expanded: Tensor = node_emb.unsqueeze(2).expand(-1, -1, seq, -1)

        # parse seq data
        # === expand indices to match feature shape
        # shape: (batch_size, node_num, seq) =>
        # (batch_size, node_num, seq, 1) =>
        # (batch_size, node_num, seq, emb)
        edge_seq_indices: Tensor = input_edge_indices.unsqueeze(-1).expand(-1, -1, -1, emb_dim)

        # === gather feature/coord
        return torch.gather(node_emb_expanded, 1, edge_seq_indices)

    @staticmethod
    def _generate_edge_coord(input_node_coord: Tensor, input_edge_indices: Tensor) -> torch.Tensor:
        coord_dim: int = input_node_coord.shape[-1]  # coord for each of the node
        seq: int = input_edge_indices.shape[-1]  # neighbours seq for each of the center node

        # === expand node feature to match indices shape
        # shape: (batch_size, node_num, node_coord_dim) =>
        # (batch_size, node_num, 1, node_coord_dim) =>
        # (batch_size, node_num, seq, node_coord_dim)
        node_coord_expanded: Tensor = input_node_coord.unsqueeze(2).expand(-1, -1, seq, -1)

        # parse seq data
        # === expand indices to match feature shape
        # shape: (batch_size, node_num, seq) =>
        # (batch_size, node_num, seq, 1) =>
        # (batch_size, node_num, seq, node_coord_dim)
        indices_coord_expanded: Tensor = input_edge_indices.unsqueeze(-1).expand(-1, -1, -1, coord_dim)

        # === gather coord
        node_seq_coord: Tensor = torch.gather(node_coord_expanded, 1, indices_coord_expanded)

        edge_coord: Tensor = node_coord_expanded - node_seq_coord

        return edge_coord

    def forward(self, x: Dict[str, Tensor], **kwargs):
        # ====== Input data
        # ============ input transform
        x_trans: Dict[str, Tensor] = {}
        for preprocess_layer in self._input_layer:
            n = preprocess_layer.prefix_name
            if n in self._input_layer_config:
                x_trans[f"{n}_emb"] = preprocess_layer(x[n])

        # ============ input hyper param
        select_edge_num = kwargs["select_edge_num"] if "select_edge_num" not in x else x["select_edge_num"]
        selected_node: Tensor = kwargs["selected_node"] if "selected_node" not in x else x["selected_node"]

        # ============ input fetch
        input_node_coord: Tensor = x["node_coord"]  # shape: (batch_size, node_num, coord_dim)
        input_node_laplace_coord: Tensor = x["laplace_coord"]  # shape: (batch_size, node_num, coord_dim)

        input_node_laplace_coord_emb: Tensor = x_trans["laplace_coord_emb"]  # shape: (batch_size, node_num, coord_dim)
        input_node_fea_emb: Tensor = x_trans["fiber_and_sheet_emb"]  # shape: (batch_size, node_num, node_feature_dim

        input_edge_indices: Tensor = x["edges_indices"]  # shape: (batch_size, node_num, seq)

        input_mat_param_emb: Tensor = x_trans["mat_param_emb"]  # shape: (batch_size, mat_param)
        input_pressure_emb: Tensor = x_trans["pressure_emb"]  # shape: (batch_size, pressure)
        input_shape_coeffs_emb: Tensor = x_trans["shape_coeffs_emb"]  # shape: (batch_size, graph_feature)
        input_time_emb: Tensor = x_trans["time_emb"]  # shape: (batch_size, time related)

        # ====== Encode global parameters theta
        global_fea = (
            input_mat_param_emb + input_pressure_emb + input_shape_coeffs_emb + input_time_emb
        )  # shape: (batch_size, theta_feature)
        global_fea = global_fea.unsqueeze(dim=-2)  # shape: (batch_size, 1, emb)
        global_fea_expanded = torch.tile(
            global_fea, (1, input_node_coord.shape[1], 1)
        )  # shape: (batch_size, node_num, emb)

        # ====== Message passing Encoder & Aggregate
        # ============ generate node emb (node emb itself)  TODO: test whether to involve the node itself
        node_emb = torch.concat(
            [input_node_laplace_coord_emb, input_node_fea_emb, global_fea_expanded], dim=-1
        )  # (batch_size, node_num, node_emb)
        center_node_emb = node_emb.unsqueeze(dim=-2).expand(
            -1, -1, select_edge_num, -1
        )  # (batch_size, node_num, 1, node_emb) => (batch_size, node_num, seq, node_emb)

        # ============ generate edge emb (agg by neighbours emb)
        selected_edge = self._random_select_edge(
            input_edge_indices, self._device, select_edge_num
        )  # shape: (batch_size, node_num, seq of edge)
        edge_node_emb = self._generate_edge_emb(node_emb, selected_edge)  # shape: (batch_size, node_num, seq, node_emb)

        # ============ generate relative coord emb (agg vertices emb at both ends + segment emb)
        edge_coord = self._generate_edge_coord(
            input_node_coord, selected_edge
        )  # (batch_size, node_num, seq, coord_emb)
        edge_laplace_coord = self._generate_edge_coord(
            input_node_laplace_coord, selected_edge
        )  # (batch_size, node_num, seq, coord_emb)

        # node_coord_emb = self.node_mlp_layer(node_coord_expanded)  # (batch_size, node_num, seq, coord_emb)
        # node_seq_coord_emb = self.node_mlp_layer(node_seq_coord)  # (batch_size, node_num, seq, coord_emb)
        edge_coord_emb = self._edge_mlp_layer(edge_coord)  # (batch_size, node_num, seq, coord_emb)
        edge_laplace_coord_emb = self._edge_laplace_mlp_layer(
            edge_laplace_coord
        )  # (batch_size, node_num, seq, coord_emb)

        coord_emb = edge_coord_emb + edge_laplace_coord_emb

        if self._message_passing_layer_config["arch"] == "attention":
            # node_emb_up = emb_concat.view(
            #     -1, emb_concat.shape[2], emb_concat.shape[3]
            # )  # shape: (batch_size * selected_node_num, seq_len, embed_dim)
            # node_emb_up = self._message_update_layer(node_emb_up)  # shape: (batch_size * node_num, seq, node_emb)
            # node_emb_up = node_emb_up.view(emb_concat.shape)  # shape: (batch_size, node_num, seq, node_emb)
            # node_emb_up = self._message_update_layer_mlp(node_emb_up)

            center_node_emb_viewed = center_node_emb[:, selected_node, :, :].view(
                -1, center_node_emb.shape[2], center_node_emb.shape[3]
            )  # shape: (batch_size * selected_node_num, seq_len, embed_dim)
            edge_node_emb_viewed = edge_node_emb[:, selected_node, :, :].view(
                -1, edge_node_emb.shape[2], edge_node_emb.shape[3]
            )  # shape: (batch_size * selected_node_num, seq_len, embed_dim)
            coord_emb_viewed = coord_emb[:, selected_node, :, :].view(
                -1, coord_emb.shape[2], coord_emb.shape[3]
            )  # shape: (batch_size * selected_node_num, seq_len, embed_dim)
            node_emb_up = self._message_update_layer(
                center_node_emb_viewed, edge_node_emb_viewed + coord_emb_viewed, edge_node_emb_viewed + coord_emb_viewed
            )  # shape: (batch_size * node_num, seq, node_emb)
            node_emb_up = node_emb_up.view(
                center_node_emb[:, selected_node, :, :].shape
            )  # shape: (batch_size, node_num, seq, node_emb)
            node_emb_up = self._message_update_layer_mlp(node_emb_up)
        else:
            emb_concat = torch.concat([center_node_emb, edge_node_emb, coord_emb], dim=-1)[:, selected_node, :, :]

            node_emb_up = self._message_update_layer(emb_concat)  # shape: (batch_size, node_num, seq, node_emb)

        node_emb_pooling = self._message_agg_pooling(node_emb_up)  # shape: (batch_size, node_num, node_emb)

        # ============ res
        z_local = node_emb[:, selected_node, :] + node_emb_pooling

        # ====== Encode global parameters theta
        # global_fea = self._theta_encode_mlp_layer(
        #     torch.concat([input_mat_param_emb, input_pressure_emb, input_shape_coeffs_emb, input_time_emb], dim=-1)
        # )  # shape: (batch_size, theta_feature)
        # global_fea = global_fea.unsqueeze(dim=-2)  # shape: (batch_size, 1, emb)
        # global_fea_expanded = torch.tile(global_fea, (1, z_local.shape[1], 1))  # shape: (batch_size, node_num, emb)

        # ====== concat local & global
        # encoding_emb = torch.concat((global_fea_expanded, z_local), dim=-1)  # shape: (batch_size, node_num, emb)

        # ====== decoder
        # make prediction for forward displacement using different decoder mlp for each dimension
        individual_mlp_predictions = [
            decode_mlp(z_local) for decode_mlp in self._decoder_layer
        ]  # shape: List[(batch_size, node_num, 1)]

        # concatenate the predictions of each individual decoder mlp
        output = dict()

        output["displacement"] = torch.concat(individual_mlp_predictions, dim=-1)  # shape: (batch_size, node_num, 1)

        return output
