import torch

def calc_iou(pred_masks: torch.Tensor, gt_masks: torch.Tensor):
    pred_masks = (pred_masks >= 0.5).float()
    gt_masks = gt_masks.float()
    # Flatten tensors
    pred_masks_flat = pred_masks.view(pred_masks.shape[0], -1)  # Shape: [N, H*W]
    gt_masks_flat = gt_masks.view(gt_masks.shape[0], -1)        # Shape: [N, H*W]
    intersection = (pred_masks_flat * gt_masks_flat).sum(dim=1)
    union = pred_masks_flat.sum(dim=1) + gt_masks_flat.sum(dim=1) - intersection
    epsilon = 1e-7
    iou = intersection / (union + epsilon) 
    return iou

def collate_fn(batch):
    batch_dict = {}
    
    for key in batch[0].keys():
        # Stack the tensors for each key
        batch_dict[key] = torch.cat([item[key] for item in batch], dim=0)

    return batch_dict
