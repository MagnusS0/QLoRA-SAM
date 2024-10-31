#!/bin/bash

seeds=(42 43 44 45 46)  # List of seeds

for seed in "${seeds[@]}"; do
    echo "Starting training with seed $seed"
    python src/train.py \
        --model_path "Zigeng/SlimSAM-uniform-77" \
        --root_dir /home/magnus/Datasets/Images/TrashCan1.0/dataset/instance_version \
        --train_annotation_file /home/magnus/Datasets/Images/TrashCan1.0/dataset/instance_version/instances_train_trashcan.json \
        --val_annotation_file /home/magnus/Datasets/Images/TrashCan1.0/dataset/instance_version/instances_val_trashcan.json \
        --output_dir ./results/seed_$seed \
        --train_batch_size 8 \
        --eval_batch_size 8 \
        --num_train_epochs 15 \
        --learning_rate 2e-4 \
        --data_loader_num_workers 16 \
        --weight_decay 10 \
        --gradient_accumulation_steps 2 \
        --seed $seed
    echo "Finished training with seed $seed"
done