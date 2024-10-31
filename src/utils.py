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
        True, 
        trace_alloc_max_entries=100000,           
        trace_alloc_record_context=True
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
    pred_mask = (pred_mask >= 0.5).float()

    intersection = torch.sum(pred_mask * gt_mask, dim=(3, 4))
    union = torch.sum(pred_mask, dim=(3, 4)) + torch.sum(gt_mask, dim=(3, 4)) - intersection
    epsilon = 1e-7

    batch_iou = intersection / (union + epsilon)
    
    return batch_iou


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, Optional[torch.Tensor]]:
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
class SamVisionAttentionSplit(SamVisionAttention, nn.Module):
    def __init__(self, config, window_size):
        super().__init__(config, window_size)
        del self.qkv
        # Separate q, k, v projections
        self.q = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.k = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.v = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self._register_load_state_dict_pre_hook(self.split_q_k_v_load_hook)

    def split_q_k_v_load_hook(self, state_dict, prefix, *args):
        keys_to_delete = []
        for key in list(state_dict.keys()):
            if "qkv." in key:
                # Split q, k, v from the combined projection
                q, k, v = state_dict[key].chunk(3, dim=0)
                # Replace with individual q, k, v projections
                state_dict[key.replace("qkv.", "q.")] = q
                state_dict[key.replace("qkv.", "k.")] = k
                state_dict[key.replace("qkv.", "v.")] = v
                # Mark the old qkv key for deletion
                keys_to_delete.append(key)
        
        # Remove old qkv keys
        for key in keys_to_delete:
            del state_dict[key]

    def forward(self, hidden_states: torch.Tensor, output_attentions=False) -> torch.Tensor:
        batch_size, height, width, _ = hidden_states.shape
        qkv_shapes = (batch_size *  self.num_attention_heads,  height * width, -1)
        query = self.q(hidden_states).reshape((batch_size,  height * width,self.num_attention_heads, -1)).permute(0,2,1,3).reshape(qkv_shapes)
        key = self.k(hidden_states).reshape((batch_size,  height * width,self.num_attention_heads, -1)).permute(0,2,1,3).reshape(qkv_shapes)
        value = self.v(hidden_states).reshape((batch_size,  height * width,self.num_attention_heads, -1)).permute(0,2,1,3).reshape(qkv_shapes)

        attn_weights = (query * self.scale) @ key.transpose(-2, -1)

        if self.use_rel_pos:
            attn_weights = self.add_decomposed_rel_pos(
                attn_weights, query, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width)
            )

        attn_weights = torch.nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = (attn_probs @ value).reshape(batch_size, self.num_attention_heads, height, width, -1)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)
        attn_output = self.proj(attn_output)

        if output_attentions:
            outputs = (attn_output, attn_weights)
        else:
            outputs = (attn_output, None)
        return outputs
    
class SamSdpaVisionAttentionSplit(SamVisionAttention, nn.Module):
    def __init__(self, config, window_size):
        super().__init__(config, window_size)
        del self.qkv
        # Separate q, k, v projections
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self._register_load_state_dict_pre_hook(self.split_q_k_v_load_hook)

    def split_q_k_v_load_hook(self, state_dict, prefix, *args):
        keys_to_delete = []
        for key in list(state_dict.keys()):
            if "qkv." in key:
                # Split q, k, v from the combined projection
                q, k, v = state_dict[key].chunk(3, dim=0)
                # Replace with individual q, k, v projections
                state_dict[key.replace("qkv.", "q.")] = q
                state_dict[key.replace("qkv.", "k.")] = k
                state_dict[key.replace("qkv.", "v.")] = v
                # Mark the old qkv key for deletion
                keys_to_delete.append(key)
        
        # Remove old qkv keys
        for key in keys_to_delete:
            del state_dict[key]
    
    def add_decomposed_rel_pos(
        self,
        query: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
        Args:
            q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
            rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
            rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
            q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
            k_size (Tuple): spatial sequence size of key k with (k_h, k_w).
        Returns:
            attn (Tensor): attention map with added relative positional embeddings.
        """
        query_height, query_width = q_size
        key_height, key_width = k_size
        relative_position_height = self.get_rel_pos(query_height, key_height, rel_pos_h)
        relative_position_width = self.get_rel_pos(query_width, key_width, rel_pos_w)

        batch_size, _, dim = query.shape
        reshaped_query = query.reshape(batch_size, query_height, query_width, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", reshaped_query, relative_position_height)
        rel_w = torch.einsum("bhwc,wkc->bhwk", reshaped_query, relative_position_width)
        rel_h = rel_h.unsqueeze(-1)
        rel_w = rel_w.unsqueeze(-2)
        rel_h = rel_h.reshape(batch_size, query_height * query_width, key_height, 1)
        rel_w = rel_w.reshape(batch_size, query_height * query_width, 1, key_width)

        return rel_h, rel_w

    def forward(self, hidden_states: torch.Tensor, output_attentions=False) -> torch.Tensor:
        batch_size, height, width, _ = hidden_states.shape
        # Apply separated projections
        query = self.query(hidden_states).reshape(batch_size, height * width, self.num_attention_heads, -1).permute(0, 2, 1, 3)
        key = self.key(hidden_states).reshape(batch_size, height * width, self.num_attention_heads, -1).permute(0, 2, 1, 3)
        value = self.value(hidden_states).reshape(batch_size, height * width, self.num_attention_heads, -1).permute(0, 2, 1, 3)
        
        # Reshape to (B * nHead, H * W, C)
        query = query.reshape(batch_size * self.num_attention_heads, height * width, -1)
        key = key.reshape(batch_size * self.num_attention_heads, height * width, -1)
        value = value.reshape(batch_size * self.num_attention_heads, height * width, -1)

        rel_h, rel_w = None, None
        if self.use_rel_pos:
            rel_h, rel_w = self.add_decomposed_rel_pos(query, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width))

        query = query.view(batch_size, self.num_attention_heads, height * width, -1)
        key = key.view(batch_size, self.num_attention_heads, height * width, -1)
        value = value.view(batch_size, self.num_attention_heads, height * width, -1)

        if self.use_rel_pos:
            rel_h = rel_h.view(batch_size, self.num_attention_heads, rel_h.size(1), rel_h.size(2), rel_h.size(3))
            rel_w = rel_w.view(batch_size, self.num_attention_heads, rel_w.size(1), rel_w.size(2), rel_w.size(3))
            attn_bias = (rel_h + rel_w).view(batch_size, self.num_attention_heads, rel_h.size(2), rel_h.size(3) * rel_w.size(4))
            attn_output = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attn_bias)
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(query, key, value)

        attn_output = attn_output.view(batch_size, self.num_attention_heads, height, width, -1).permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)

        attn_output = self.proj(attn_output)

        if output_attentions:
            # For output_attentions, calculate the attention weights
            attn_weights = (query @ key.transpose(-2, -1)) * self.scale
            if attn_bias is not None:
                attn_weights = attn_weights + attn_bias
            attn_weights = F.softmax(attn_weights, dim=-1)
            outputs = (attn_output, attn_weights)
        else:
            outputs = (attn_output, None)

        return outputs