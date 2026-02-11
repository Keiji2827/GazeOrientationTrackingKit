#!/bin/bash

IMAGE_PATH="data/preprocessed/lab/1013_1/Camera_1_0/"
IMAGE_NAME="000123.jpg"

python3 -m models.tools.oneshot_inference \
  --target_image_path "${IMAGE_PATH}/${IMAGE_NAME}" \
  --image_dir "${IMAGE_PATH}" \
  --annotation_path "${IMAGE_PATH}/annotation.pickle" \
  --model_checkpoint "output/checkpoint-10-120000/state_dict.bin" \
  --output_jpeg "output/images/overlay_${IMAGE_NAME}" \
  --device cuda

