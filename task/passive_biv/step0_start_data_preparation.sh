#!/bin/bash

# Configure and launch the passive BiV data-preparation stage.
export TASK_NAME="passive_biv"

export PROJECT_PATH="$(cd `dirname $0`/../../; pwd)"
echo "project root path: ${PROJECT_PATH}"

export TASK_TYPE="data_preparation"

export CONFIG_PATH="task/${TASK_NAME}/data_preparation/config/data_config.yaml"

# Optionally cap Numba worker threads for shared compute environments.
#export NUMBA_NUM_THREADS=16
#echo "${NUMBA_NUM_THREADS}"

sh "${PROJECT_PATH}/common/sbin/main_process.sh"
