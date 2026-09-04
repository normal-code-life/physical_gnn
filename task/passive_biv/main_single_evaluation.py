"""Evaluate selected passive BiV cases and save aggregate error measurements."""

import argparse
import os
import random
import sys
from typing import List

import numpy as np
import torch
from torch import nn

from pkg.utils.other_utils import io
from pkg.utils.other_utils.logs import init_logger
from task.passive_biv.model_eval.eval.datasets_eval_hdf5 import FEHeartSimSageEvaluation
from task.passive_biv.utils.utils import import_data_config

logger = init_logger("SINGLE_CASE_EVAL")


def load_model(repo_path: str, model_path: str, device: str) -> nn.Module:
    """Load and prepare model for evaluation.

    Returns:
        nn.Module: Loaded PyTorch model in evaluation mode
    """
    path = f"{repo_path}/{model_path}"
    loaded_model = torch.load(path, map_location=torch.device("cpu"))

    if isinstance(loaded_model, torch.nn.DataParallel):
        loaded_model = loaded_model.module

    if hasattr(loaded_model, "device"):
        loaded_model.device = device

    loaded_model.eval()

    return loaded_model


if __name__ == "__main__":
    cur_path = os.path.abspath(sys.argv[0])
    task_dir = io.get_repo_path(cur_path)
    sys.argv.extend(
        [
            "--repo_path",
            f"{task_dir}",
            "--task_name",
            "passive_biv",
            "--dataset_name",
            "passive_biv" "--model_name",
            "fe_heart_sim_sage",
            "--task_type",
            "model_eval",
        ]
    )

    parser = argparse.ArgumentParser(description="Evaluation")

    parser.add_argument(
        "--config_path",
        type=str,
        default="task/passive_biv/model_eval/config/eval_config.yaml",
        help="config path location",
    )

    args: (argparse.Namespace, List[str]) = parser.parse_known_args()

    config = import_data_config(args[0].config_path)

    model = load_model(config["repo_path"], config["model_path"], config["device"])
    logger.info("=== Print Model Structure ===")
    logger.info(model)

    # str_summary = summary_model(
    #     model,
    #     inputs,
    #     show_input=True,
    #     show_hierarchical=True,
    #     # print_summary=model_summary["print_summary"],
    #     max_depth=999,
    #     show_parent_layers=True,
    # )

    # logger.info(str_summary)

    l1_results = []
    l2_results = []
    sample_l1_results = []
    sample_l2_results = []
    results_source = []
    results_pred = []

    evaluation = FEHeartSimSageEvaluation(config)

    def preprocess(x):
        """Return model inputs unchanged; kept as an extension hook."""
        return x

    for idx in [
        2,
    ]:
        inputs, labels = evaluation.generate_single_inputs(idx - 1)

        inputs = preprocess(inputs)

        _, node_num, _ = inputs["edges_indices"].shape

        select_node = random.sample(list(range(node_num)), config["selected_node"])

        with torch.no_grad():
            output = model(
                inputs,
                selected_node=list(range(node_num)),
                select_edge_num=config["select_edge_num"],
            )

            output_norm = evaluation.convert_outputs(output)
            l1_results.append(evaluation.calc_l1_loss(output_norm, labels["displacement"]).numpy())
            l2_results.append(evaluation.calc_l2_loss(output_norm, labels["displacement"]).numpy())
            results_source.append(labels["displacement"][:, select_node, :].numpy())
            results_pred.append(output_norm[:, select_node, :].numpy())

            sample_l1_results.append(
                evaluation.sample_l1_loss(
                    output_norm[:, select_node, :], labels["displacement"][:, select_node, :]
                ).numpy()
            )
            sample_l2_results.append(
                evaluation.sample_l2_loss(
                    output_norm[:, select_node, :], labels["displacement"][:, select_node, :]
                ).numpy()
            )

            evaluation.save_single_outputs(output_norm, idx - 1)

        print(idx, "job done")

    evaluation.save_numpy_outputs("l1_results", l1_results)
    evaluation.save_numpy_outputs("l2_results", l2_results)
    evaluation.save_numpy_outputs("sample_l1_results", sample_l1_results)
    evaluation.save_numpy_outputs("sample_l2_results", sample_l2_results)
    evaluation.save_numpy_outputs("results_source", results_source)
    evaluation.save_numpy_outputs("results_pred", results_pred)

    print(f"l1_results: {np.median(l1_results)}")
    print(f"l2_results: {np.median(l2_results)}")
