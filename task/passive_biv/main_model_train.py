"""Command-line entry point for passive BiV model training."""

import argparse
import os
import sys
from typing import List

from pkg.utils.other_utils import io
from task.passive_biv.fe_heart_sim_sage.train.model import FEHeartSimSageTrainer

if __name__ == "__main__":
    # Supply project defaults when the script is launched without runner arguments.
    if len(sys.argv) <= 1 or "repo_path" not in sys.argv[1]:
        cur_path = os.path.abspath(sys.argv[0])
        task_dir = io.get_repo_path(cur_path)

        sys.argv.extend(
            [
                "--repo_path",
                f"{task_dir}",
                "--task_name",
                "passive_biv",
                "--model_name",
                "fe_heart_sim_sage",
                "--config_name",
                "train_config",
                "--task_type",
                "model_train",
            ]
        )

    parser = argparse.ArgumentParser(description="Model Selection")

    parser.add_argument("--model_name", type=str, default="fe_heart_sim_sage", help="model name")

    args: (argparse.Namespace, List[str]) = parser.parse_known_args()

    # Dispatch to the trainer selected on the command line.
    if args[0].model_name == "fe_heart_sim_sage":
        model = FEHeartSimSageTrainer()
    else:
        raise ValueError("please pass the right model name")

    model.train()
