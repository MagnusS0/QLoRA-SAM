from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.sam.modeling_sam import SamVisionAttention
import pickle

def memory_runner(path, fn, *args, **kwargs):
    print("Start memory recording")
    torch.cuda.synchronize()
    torch.cuda.memory._record_memory_history(
        enabled=True,
        trace_alloc_max_entries=100000,           
        trace_alloc_record_context=True,
    )
    result = fn(*args, **kwargs)
    torch.cuda.synchronize()
    snapshot = torch.cuda.memory._snapshot()
    print("Finish memory recording")
    with open(path, 'wb') as f:
        pickle.dump(snapshot, f)
    # Use to convert pickle file into html
    # python torch/cuda/_memory_viz.py trace_plot <snapshot>.pickle -o <snapshot>.html
    return result

def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    """
    Calculate the Intersection over Union (IoU) between predicted and ground truth masks.

    Args:
        pred_mask (torch.Tensor): Predicted mask tensor.
        gt_mask (torch.Tensor): Ground truth mask tensor.

    Returns:
        torch.Tensor: IoU scores for each sample in the batch.
    """
    pred_mask = (pred_mask >= 0.5).float()

    intersection = torch.sum(pred_mask * gt_mask, dim=(3, 4))
    union = torch.sum(pred_mask, dim=(3, 4)) + torch.sum(gt_mask, dim=(3, 4)) - intersection
    epsilon = 1e-7

    batch_iou = intersection / (union + epsilon)
    
    return batch_iou


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, Optional[torch.Tensor]]:
    """
    Custom collate function for DataLoader to handle variable size inputs.

    Args:
        batch (List[Dict[str, torch.Tensor]]): List of samples from the dataset.

    Returns:
        Dict[str, Optional[torch.Tensor]]: Batched data.
    """
    batch_dict = {
        'pixel_values': torch.stack([item['pixel_values'].squeeze() for item in batch]),
        'original_sizes': torch.stack([item['original_sizes'] for item in batch]),
        'reshaped_input_sizes': torch.stack([item['reshaped_input_sizes'] for item in batch])
    }

    max_masks = max(item['labels'].shape[0] for item in batch)
    max_height = max(item['labels'].shape[1] for item in batch)
    max_width = max(item['labels'].shape[2] for item in batch)
    
    optional_inputs = ['input_boxes', 'input_labels', 'input_points']

    def pad_tensor(tensor: torch.Tensor, target_shape: List[int], pad_value: int = -1) -> torch.Tensor:
        pad_sizes = []
        for src, tgt in zip(reversed(tensor.shape), reversed(target_shape)):
            pad = max(0, tgt - src)
            pad_sizes.extend([0, pad])  # Pad only at the end of each dimension
        return torch.nn.functional.pad(tensor, pad_sizes, mode='constant', value=pad_value)

    # Handle 'labels' separately
    batch_dict['labels'] = torch.stack([
        pad_tensor(item['labels'], [max_masks, max_height, max_width], pad_value=0)
        for item in batch
    ])

    # Handle optional inputs
    for key in optional_inputs:
        if key in batch[0]:
            if key == 'input_boxes':
                target_shape = [1, max_masks, 4]
            elif key == 'input_labels':
                target_shape = [1, 1, max_masks]
            elif key == 'input_points':
                target_shape = [1, max_masks, 2]
            else:
                continue
                
            batch_dict[key] = torch.stack([
                pad_tensor(item[key], target_shape).squeeze(0)  # Remove extra batch dimension if necessary
                for item in batch
            ])
        else:
            batch_dict[key] = None

    return batch_dict