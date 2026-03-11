#!/bin/bash

# bash run_oneshot.sh
# Run as bash


SAERCH_DIR="data/preprocessed/courtyard/002"
NUM_SELECT=1000

# ------------------------------------------
# Collect all jpg files recursively
# ------------------------------------------
mapfile -t ALL_FILES < <(find "$SAERCH_DIR" -type f -iname "*.jpg")


TOTAL=${#ALL_FILES[@]}

if [ "$TOTAL" -eq 0 ]; then
    echo "No jpg files found."
    exit 1
fi

if [ "$TOTAL" -lt "$NUM_SELECT" ]; then
    echo "Warning: Only $TOTAL files found. Selecting all."
    NUM_SELECT=$TOTAL
fi

# ------------------------------------------
# Shuffle and select
# ------------------------------------------
mapfile -t SELECTED < <(
    printf "%s\n" "${ALL_FILES[@]}" | shuf | head -n "$NUM_SELECT"
)

# ------------------------------------------
# Create space-separated string
# ------------------------------------------
IMAGE_PATH="${SELECTED[*]}"

echo "Selected $NUM_SELECT files."
echo

#IMAGE_PATH="data/preprocessed/lab/1013_1/Camera_1_0/000155.jpg"
#IMAGE_PATH+=" data/preprocessed/living_room/005/Camera_12_3/000155.jpg"
#IMAGE_PATH+=" data/preprocessed/courtyard/003/Camera_7_0/000155.jpg"
#IMAGE_PATH+=" data/preprocessed/kitchen/1022_4/Camera_1_6/000122.jpg"
#IMAGE_PATH+=" data/preprocessed/library/1029_2/Camera_8_2/001224.jpg"

#echo "$IMAGE_PATH"


python3 -m models.tools.oneshot_inference \
  --target_image_path ${IMAGE_PATH} \
  --n_frames 7 \
  --model_checkpoint "output/checkpoint-10-120000_n7/state_dict.bin" \
  --output_path "output_images/courtyard/" \
  --device cuda

