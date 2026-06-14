from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from pkg.train.model.base_model import BaseModule
from pkg.utils.other_utils.logs import init_logger

logger = init_logger("MLP_LAYER_LN")


class MLPLayerBase(BaseModule):
    def __init__(self, config: Dict, **kwargs) -> None:
        super().__init__(config, **kwargs)

        self._layer_name = f"{self._prefix_name}_mlp"

        self._unit_sizes: List[int] = []
        if isinstance(config["unit_sizes"], list):
            # WARN: layer_sizes should contain the first input layer and final layer's output dim
            self._unit_sizes = config["unit_sizes"]
        else:
            raise ValueError("the 'unit_sizes' should be a list, and should contain the final layer's output size")

        self._batch_norm = config.get("batch_norm", False)
        self._layer_norm = config.get("layer_norm", False)
        self._activation = config.get(
            "activation", None
        )  # by default, the last layer will not have the activation func
        self._init_func = config.get("init_func", "xavier_uniform")
        self._init_weight_file_path = config.get("init_weight_file_path", None)  # if not None, weight will be assigned

        self._mlp_layers: nn.Sequential = nn.Sequential()
        self._init_graph()

    def get_config(self) -> Dict:
        base_config = super().get_config()

        mlp_config = {
            "unit_sizes": self._unit_sizes,
            "layer_name": self._layer_name,
            "batch_norm": self._batch_norm,
            "layer_norm": self._layer_norm,
            "activation": self._activation,
            "init_weight_file_path": self._init_weight_file_path,
        }

        return {**base_config, **mlp_config}

    def _init_graph(self) -> None:
        raise NotImplementedError("please implement this method")

    def _init_fc(
        self,
        cur_layer_name: str,
        input_unit_size: int,
        output_unit_size: int,
    ) -> None:
        fc = nn.Linear(input_unit_size, output_unit_size)

        if self._init_weight_file_path:
            with open(self._init_weight_file_path, "rb") as file:
                weight_init_dict = np.load(file, allow_pickle=True).item()

            if cur_layer_name in weight_init_dict:
                fc.weight = nn.Parameter(torch.tensor(weight_init_dict[cur_layer_name]).t())
                logger.info(f"init {cur_layer_name} model layer from {self._init_weight_file_path}")
            else:
                raise ValueError(f"error, we don't have this layer {cur_layer_name}")
        else:
            if self._init_func == "xavier_uniform":
                nn.init.xavier_uniform_(fc.weight)
            elif self._init_func == "xavier_normal":
                nn.init.xavier_normal_(fc.weight)
            else:
                raise Exception(f"please define the init_func correctly, currently init_func={self._init_func}")

        nn.init.zeros_(fc.bias)
        self._mlp_layers.add_module(cur_layer_name, fc)

    def forward(self, x):
        return self._mlp_layers(x)
