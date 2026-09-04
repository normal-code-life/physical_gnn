#!/bin/bash

# check CUDA info
#echo "${CUDA_PATH}"
#echo "${CUDA_HOME}"

export PYTHONPATH=${PYTHONPATH}:${PROJECT_PATH}

# Resolve the task entry point from the task mode supplied by the launcher.
TASK_PATH=${PROJECT_PATH}/task/${TASK_NAME}

echo "task_path: ${TASK_PATH}"
echo "task_type: ${TASK_TYPE}"

if [ "${TASK_TYPE}" = "data_preparation" ]; then
  echo "config_path: ${CONFIG_PATH}"
  args="--config_path ${CONFIG_PATH}"

  file_name="main_data_preparation"

elif [ "${TASK_TYPE}" = "model_train" ]; then
  args="--repo_path ${PROJECT_PATH} --task_name ${TASK_NAME} --model_name ${MODEL_NAME} --config_name ${CONFIG_NAME}  --task_type ${TASK_TYPE}"

  file_name="main_model_train"

elif [ "${TASK_TYPE}" = "model_evaluation" ]; then
  file_name="main_model_evaluation"

elif [ "${TASK_TYPE}" = "main_single_evaluation" ]; then
  file_name="main_single_evaluation"

elif [ "${TASK_TYPE}" = "main_single_evaluation_v2" ]; then
  file_name="main_single_evaluation_v2"

else
  echo "Invalid TASK_TYPE: ${TASK_TYPE}"
  exit 1
fi

# Forward the mode-specific arguments to the selected Python entry point.
python "${TASK_PATH}"/"${file_name}".py ${args}
