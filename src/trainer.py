from transformers import Trainer
from utils import calc_iou
import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp
from monai.losses import DiceFocalLoss

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize Loss Functions
        self.dice_focal_loss = DiceFocalLoss(sigmoid=True, squared_pred=True, reduction='mean', lambda_focal = 20.0)
        self.mse_loss = nn.MSELoss()
        
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom compute_loss that includes Focal Loss, Dice Loss, and IoU-based Loss.
        """
        labels = inputs.pop('labels').float().to(self.args.device) 
        pixel_values = inputs['pixel_values']
        input_boxes = inputs.get('input_boxes')
        input_labels = inputs.get('input_labels')
        input_points = inputs.get('input_points')

        # Forward pass
        outputs = model(
            pixel_values=pixel_values,
            input_boxes=input_boxes,
            input_labels=input_labels,
            input_points=input_points,
            multimask_output=False,
        )
        iou_predictions = outputs.iou_scores
        pred_masks = outputs.pred_masks

        pred_masks = pred_masks.squeeze(2)

        assert pred_masks.shape == labels.shape, f"Shape mismatch: pred_masks {pred_masks.shape}, labels {labels.shape}"

        # Add channel dimension for loss computation
        pred_masks = pred_masks.unsqueeze(2) 
        labels = labels.unsqueeze(2)     
        
        loss_dice_focal = self.dice_focal_loss(pred_masks, labels) 
        
        # Compute IoU for each mask in batch and compute IoU-based loss
        batch_iou = calc_iou(pred_masks, labels) 
        loss_iou = self.mse_loss(iou_predictions, batch_iou)  # MSE between predicted and true IoU scores
        
        loss_total = loss_dice_focal + loss_iou

        return (loss_total, outputs) if return_outputs else loss_total
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """
        Custom evaluation loop that computes metrics similar to original code without interpolation and flattening.
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        model = self.model
        model.eval()

        total_iou = 0.0
        total_f1 = 0.0
        total_loss = 0.0
        total_samples = 0

        for _, inputs in enumerate(eval_dataloader):
            labels = inputs.pop('labels').float().to(self.args.device)
            pixel_values = inputs['pixel_values']
            num_images = pixel_values.size(0)
            input_boxes = inputs.get('input_boxes')
            input_labels = inputs.get('input_labels')
            input_points = inputs.get('input_points')

            with torch.no_grad():
                outputs = model(
                    pixel_values=pixel_values,
                    input_boxes=input_boxes,
                    input_labels=input_labels,
                    input_points=input_points,
                    multimask_output=False,
                )
                pred_masks = outputs.pred_masks
                pred_masks = pred_masks.squeeze(2)

                assert pred_masks.shape == labels.shape, f"Shape mismatch: pred_masks {pred_masks.shape}, labels {labels.shape}"

                # Add channel dimension for loss computation
                pred_masks = pred_masks.unsqueeze(2) 
                labels = labels.unsqueeze(2)     


                tp, fp, fn, tn = smp.metrics.get_stats(
                pred_masks,
                labels.int(),
                mode='binary',
                threshold=0.5,
            )

                # Compute IoU and F1
                batch_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")

                # Accumulate metrics
                total_iou += batch_iou.item() * num_images
                total_f1 += batch_f1.item() * num_images
                total_samples += num_images

                # Accumulate loss
                loss = self.compute_loss(model, {
                    'pixel_values': pixel_values,
                    'input_boxes': input_boxes,
                    'input_labels': input_labels,
                    'labels': labels.squeeze(2)  # Revert labels back to original
                }, return_outputs=False)
                total_loss += loss.item() * num_images

                print(
                    f'Val: Mean IoU: [{batch_iou.item():.4f}] -- Mean F1: [{batch_f1.item():.4f}]'
                )

        # Compute Final Metrics
        final_iou = total_iou / total_samples if total_samples > 0 else 0.0
        final_f1 = total_f1 / total_samples if total_samples > 0 else 0.0
        average_loss = total_loss / total_samples if total_samples > 0 else 0.0

        # Prepare metrics dictionary
        metrics = {
            f"{metric_key_prefix}_mean_iou": final_iou,
            f"{metric_key_prefix}_mean_f1": final_f1,
            f"{metric_key_prefix}_loss": average_loss
        }

        # Log metrics using Trainer's built-in logging
        self.log(metrics)

        return metrics