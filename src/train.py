import argparse
import os
from transformers import TrainingArguments
from trainer import CustomTrainer
from dataset import COCODataset
from transformers import SamProcessor, SamModel
from lora import configure_lora_model
from utils import collate_fn

def main(args):
    # Initialize the processor
    processor = SamProcessor.from_pretrained(args.model_path)

    # Initialize the model
    model = configure_lora_model(args.model_path)

    # Prepare datasets
    train_ds = COCODataset(
        root_dir=os.path.join(args.root_dir, 'train'),
        annotation_file=args.train_annotation_file,
        processor=processor
    )
    val_ds = COCODataset(
        root_dir=os.path.join(args.root_dir, 'val'),
        annotation_file=args.val_annotation_file,
        processor=processor
    )
    if args.test_annotation_file is not None:
        test_ds = COCODataset(
            root_dir=os.path.join(args.root_dir, 'test'),
            annotation_file=args.test_annotation_file,
            processor=processor
        )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        dataloader_num_workers=args.data_loader_num_workers,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type='cosine',
        warmup_steps=args.warmup_steps,
        optim='lion_8bit',
        weight_decay=args.weight_decay,
        bf16=True,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy='steps',
        load_best_model_at_end=True,
        metric_for_best_model="eval_mean_iou",
        greater_is_better=True,
        report_to='tensorboard',
        save_total_limit=args.save_total_limit,
    )

    # Initialize the Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,  # Collation is handled by the DataLoader
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    if args.test_annotation_file is not None:
        trainer.test_dataset = test_ds

    # Start training
    trainer.train()

    # Evaluate the model on the test set
    if args.test_annotation_file is not None:
        trainer.evaluate(test_dataset=test_ds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the COCO dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Hugging Face model.")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory of the COCO dataset.")
    parser.add_argument("--train_annotation_file", type=str, required=True, help="Path to the training annotation file.")
    parser.add_argument("--val_annotation_file", type=str, required=True, help="Path to the validation annotation file.")
    parser.add_argument("--test_annotation_file", type=str, required=False, default=None, help="Path to the test annotation file.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for model checkpoints and logs.")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size per device.")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Evaluation batch size per device.")
    parser.add_argument("--data_loader_num_workers", type=int, default=4, help="Number of workers for the DataLoader.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps for learning rate scheduler.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer.")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory for TensorBoard logs.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_steps", type=int, default=500, help="Run evaluation every X steps.")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Limit the total amount of checkpoints.")
    args = parser.parse_args()
    main(args)
