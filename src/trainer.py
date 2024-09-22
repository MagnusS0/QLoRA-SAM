from torch.utils.data import DataLoader
from transformers import Trainer
from losses import FocalLoss, DiceLoss
from utils import calc_iou
import torch
import torch.nn as nn
import torch.nn.functional as F
import evaluate

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss_fn = FocalLoss()
        self.dice_loss_fn = DiceLoss()
        self.mse_loss_fn = nn.MSELoss()
        self.metric = evaluate.load("mean_iou")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels').float()  # Shape: [batch_size, H, W]
        outputs = model(**inputs)
        pred_masks = outputs.pred_masks  # Shape: [batch_size, num_masks, H_pred, W_pred]
        iou_predictions = outputs.iou_scores  # Shape: [batch_size, num_masks]

        if pred_masks.dim() == 5:  # If shape is [batch_size, 1, num_masks, H, W]
            pred_masks = pred_masks.squeeze(1)  # Remove the singleton dimension

        # Resize pred_masks to match labels
        pred_masks = F.interpolate(pred_masks, size=labels.shape[-2:], mode='bilinear', align_corners=False)
        batch_size, num_masks, H, W = pred_masks.shape

        # Expand labels to match pred_masks
        labels = labels.unsqueeze(1).repeat(1, num_masks, 1, 1)  # Shape: [batch_size, num_masks, H, W]

        # Flatten tensors
        pred_masks_flat = pred_masks.reshape(-1, H, W)
        labels_flat = labels.reshape(-1, H, W)
        iou_predictions_flat = iou_predictions.reshape(-1)

        # Compute IoU for each mask
        batch_iou = calc_iou(pred_masks_flat, labels_flat)  # Returns a tensor of IoUs

        # Compute losses
        loss_focal = self.focal_loss_fn(pred_masks_flat, labels_flat)
        loss_dice = self.dice_loss_fn(pred_masks_flat, labels_flat)
        loss_iou = F.mse_loss(iou_predictions_flat, batch_iou)

        # Total Loss
        loss = 20.0 * loss_focal + loss_dice + loss_iou

        return (loss, outputs) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        model = self.model
        model.eval()
        total_loss = 0
        total_samples = 0

        for inputs in eval_dataloader:
            labels = inputs.pop('labels').float()
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            labels = labels.to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                pred_masks = outputs.pred_masks

                if pred_masks.dim() == 5:  # If shape is [batch_size, 1, num_masks, H, W]
                    pred_masks = pred_masks.squeeze(1) 

                # Resize pred_masks to match labels
                pred_masks = F.interpolate(pred_masks, size=labels.shape[-2:], mode='bilinear', align_corners=False)

                # Apply sigmoid and threshold
                pred_masks = pred_masks.sigmoid()
                preds = (pred_masks > 0.5).float()

                # Flatten batch and mask dimensions
                batch_size, num_masks, H, W = preds.shape
                preds = preds.view(batch_size * num_masks, H, W)
                labels_expanded = labels.unsqueeze(1).repeat(1, num_masks, 1, 1)
                labels_flat = labels_expanded.view(batch_size * num_masks, H, W)

                # Compute metrics
                self.metric.add_batch(predictions=preds.cpu().numpy(), references=labels_flat.cpu().numpy())

        result = self.metric.compute(num_labels=2, ignore_index=None, reduce_labels=False)
        mean_iou = result['mean_iou']
        print(f"Validation Mean IoU: {mean_iou:.4f}")

        # Prepare the metrics dictionary
        metrics = {f"{metric_key_prefix}_mean_iou": mean_iou}

        # Log metrics
        self.log(metrics)
        return metrics