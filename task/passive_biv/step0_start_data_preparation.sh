#!/bin/bash

export TASK_NAME="passive_biv"

export PROJECT_PATH="$(cd `dirname $0`/../../; pwd)"
echo "project root path: ${PROJECT_PATH}"

export TASK_TYPE="data_preparation"

export CONFIG_PATH="task/${TASK_NAME}/data_preparation/config/data_config.yaml"

## setup NUMBA info
#export NUMBA_NUM_THREADS=16
#echo "${NUMBA_NUM_THREADS}"

sh "${PROJECT_PATH}/common/sbin/main_process.sh"
