import csv
import os
import time
import torch
import argparse
from transformers import SamProcessor, SamModel, TrainingArguments
from dataset import COCODataset
from trainer import CustomTrainer
from utils import collate_fn

def evaluate_on_test_set(trainer, test_dataset, model_name):
    """
    Evaluate the model on the test dataset and log metrics.

    Args:
        trainer: The trainer instance.
        test_dataset: The dataset to evaluate on.
        model_name (str): Name of the model for logging purposes.
    """
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()
    metrics = trainer.evaluate(eval_dataset=test_dataset)
    memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)
    elapsed_time = time.time() - start_time

    print(f"Evaluation metrics: {metrics}")
    print(f"Elapsed time: {elapsed_time:.2f} s")
    print(f"Memory usage: {memory_usage:.2f} MB")

    # Log the results
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    log_entry = [timestamp, model_name, metrics, memory_usage]
    log_file = "logs/model_benchmark.csv"
    with open(log_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(log_entry)

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Test SAM model performance')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model for logging')
    parser.add_argument('--dataset_root_dir', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--val_annotation_file', type=str, default=None, help='Path to the validation annotation file')
    args = parser.parse_args()

    # Logging setup
    log_file = "logs/model_benchmark_log.csv"
    fields = ["timestamp", "model_name", "metrics", "memory_usage (MB)"]
    if not os.path.exists("logs"):
        os.makedirs("logs")
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write headers only if the file didn't exist before
        if not file_exists:
            writer.writerow(fields)

    # Load the trained model
    model = SamModel.from_pretrained(
        args.model_path,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16
    )

    # Create the SamProcessor
    try:
        processor = SamProcessor.from_pretrained(
            args.model_path
        )
    except:
        processor = SamProcessor(
            "facebook/sam-vit-base"
        )


    # Load the test dataset
    val_annotation_file = args.val_annotation_file

    test_dataset = COCODataset(
        root_dir=os.path.join(args.dataset_root_dir, 'val'),
        annotation_file=val_annotation_file,
        processor=processor,
        no_prompt=False,
        dtype=torch.bfloat16
    )

    # Load the training arguments
    training_args = TrainingArguments(
        output_dir="output",
        per_device_eval_batch_size=1,
        dataloader_num_workers=16,
        report_to='none',
    )

    # Create a trainer instance
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        eval_dataset=test_dataset
    )

    # Evaluate the model
    evaluate_on_test_set(trainer, test_dataset, args.model_name)

if __name__ == "__main__":
    main()
