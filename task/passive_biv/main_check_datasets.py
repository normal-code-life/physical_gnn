"""Manually inspect passive BiV dataset iteration order and throughput."""

import time

from torch.utils.data import DataLoader

from common.constant import TRAIN_NAME
from task.passive_biv.fe_heart_sim_sage.train.datasets_train_hdf5 import FEHeartSimSageTrainDataset
from task.passive_biv.utils.utils import import_data_config

if __name__ == "__main__":
    data_config = import_data_config("task/passive_biv/data/config/data_config.yaml")
    data_config["data_type"] = TRAIN_NAME

    train_data = FEHeartSimSageTrainDataset(data_config)

    train_data_loader = DataLoader(
        dataset=train_data,
        batch_size=data_config.get("batch_size", 1),
        num_workers=1,
        prefetch_factor=None,
    )

    start_time = time.time()

    s = 0
    for i in range(4):
        for inputs, labels in train_data_loader:
            s += 1
            # for i in inputs:
            #     print(i, inputs[i].shape)
            # for i in labels:
            #     print(i, labels[i].shape)

            print(inputs["index"].item())

        print(f"{i}: {time.time() - start_time}s")
