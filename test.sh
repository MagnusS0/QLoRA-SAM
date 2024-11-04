#!/bin/bash

set -e

# Define an array of model configurations
# Each entry is a space-separated string: "MODEL_PATH MODEL_NAME"
declare -a models=(
    "results/seed_0/qlora_model lora_seed_0"
)

DATASET_ROOT_DIR="/home/magnus/Datasets/Images/TrashCan1.0/dataset/instance_version"
ANNOTATION_FILE="/home/magnus/Datasets/Images/TrashCan1.0/dataset/instance_version/instances_val_trashcan.json"

# Loop through each model configuration and run the test
for model in "${models[@]}"; do
    read -r MODEL_PATH MODEL_NAME <<< "$model"
    echo "Testing model: $MODEL_NAME"
    echo "Model path: $MODEL_PATH"
    echo "Dataset root directory: $DATASET_ROOT_DIR"
    
    python src/test.py \
        --model_path "$MODEL_PATH" \
        --model_name "$MODEL_NAME" \
        --dataset_root_dir "$DATASET_ROOT_DIR" \
        --val_annotation_file "$ANNOTATION_FILE"

    echo "Completed testing model: $MODEL_NAME"
    echo "----------------------------------------"
done

echo "All tests completed!"
