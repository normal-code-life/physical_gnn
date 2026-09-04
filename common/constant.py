"""Shared string constants used by data, training, and evaluation workflows."""

# Task modes.
DATA_PREPARATION = "data_preparation"
MODEL_TRAIN = "model_train"
MODEL_EVALUATION = "model_evaluation"

# Dataset splits.
TRAIN_NAME = "train"
VALIDATION_NAME = "validation"
TEST_NAME = "test"
UNKNOWN_NAME = "unknown"

# Statistical summary keys.
MAX_VAL = "max_val"
MIN_VAL = "min_val"
MEAN_VAL = "mean_val"
STD_VAL = "std_val"
MEDIAN_VAL = "median_val"
PERC_10_VAL = "10_percentile_val"
PERC_25_VAL = "25_percentile_val"
PERC_75_VAL = "75_percentile_val"
PERC_90_VAL = "90_percentile_val"
PERC_95_VAL = "95_percentile_val"
PERC_99_VAL = "99_percentile_val"

# Operating-system identifiers.
DARWIN = "Darwin"

# Supported storage formats.
HDF5 = "hdf5"
TFRecord = "tfrecord"
