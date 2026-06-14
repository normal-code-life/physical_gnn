"""Dataset Generation Module.

This module provides utilities for generating and splitting datasets for machine learning tasks.

The main functionality includes:

- Splitting datasets into training and validation sets
- Generating random indices for dataset splits
- Managing dataset indices with customizable start points

Key Functions:
    split_dataset_indices: Splits dataset indices into train/validation sets
"""

from typing import Dict

import numpy as np

from common.constant import TRAIN_NAME, VALIDATION_NAME


def split_dataset_indices(start_index: int, total_sample_size: int, train_split_ratio: float) -> Dict[str, np.ndarray]:
    """Splits a dataset into training and validation sets by randomly shuffling and dividing the indices.

    This function takes a directory of samples and splits them into training and validation sets
    based on the provided train_split_ratio. It generates random indices for the split to ensure
    the data distribution is random between sets.

    Parameters:
    ----------
    start_index: int
        indices start value
    total_sample_size : int
        total sample size including train and test data. This function will split the dataset based on
        the sample size.
    train_split_ratio : float
        Ratio of samples to use for training, must be between 0 and 1. For example,
        0.8 means 80% of samples will be used for training and 20% for validation.

    Returns:
    -------
    Dict[str, np.ndarray]
        A dictionary containing two keys:
        - TRAIN_NAME: np.ndarray of indices for training samples
        - VALIDATION_NAME: np.ndarray of indices for validation samples

    Example:
    -------
    >>> split = split_dataset_indices(1000, 0.8)
    >>> train_indices = split[TRAIN_NAME]
    >>> validation_indices = split[VALIDATION_NAME]
    """
    # Calculate number of training samples based on split ratio
    train_sample_size: int = int(total_sample_size * train_split_ratio)

    # Generate shuffled indices for the entire dataset
    sample_indices: np.ndarray = np.arange(start_index, start_index + total_sample_size)
    np.random.shuffle(sample_indices)

    # Split indices into training and validation sets
    train_indices: np.ndarray = sample_indices[:train_sample_size]
    validation_indices: np.ndarray = sample_indices[train_sample_size:]

    return {TRAIN_NAME: sorted(train_indices), VALIDATION_NAME: sorted(validation_indices)}
