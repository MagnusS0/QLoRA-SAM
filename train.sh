#!/bin/bash

python src/train.py \
    --model_path Zigeng/SlimSAM-uniform-77 \
    --root_dir /home/magnus/Datasets/Images/TrashCan1.0/dataset/instance_version \
    --train_annotation_file /home/magnus/Datasets/Images/TrashCan1.0/dataset/instance_version/instances_train_trashcan.json \
    --val_annotation_file /home/magnus/Datasets/Images/TrashCan1.0/dataset/instance_version/instances_val_trashcan.json\
    --output_dir ./results \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --data_loader_num_workers 16 \