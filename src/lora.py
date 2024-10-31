from transformers import BitsAndBytesConfig
from transformers import SamModel
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import SamVisionAttentionSplit, SamSdpaVisionAttentionSplit
from transformers.models.sam import modeling_sam



def configure_lora_model(model_path: str, quant: bool = True, train_prompt: bool = False) -> SamModel:
    # Quantization configuration
    config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
    # modeling_sam.SamVisionAttention = SamVisionAttentionSplit
    # modeling_sam.SamSdpaVisionAttention = SamSdpaVisionAttentionSplit
    
    if quant:
        model = SamModel.from_pretrained(model_path, quantization_config=config, ignore_mismatched_sizes=True, attn_implementation="sdpa", low_cpu_mem_usage=True)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = SamModel.from_pretrained(model_path, attn_implementation="sdpa", torch_dtype=torch.bfloat16)
        model.train()

    # Lora configuration
    config = LoraConfig(
    task_type="mask-generation",
    r=128,
    lora_alpha=256,
    target_modules=["qkv", "q_proj", "v_proj"],
    use_dora=True,
)
    if train_prompt:
        for name, param in model.named_parameters():
            if 'prompt_encoder' in name:
                param.requires_grad = True

    model = get_peft_model(model, config)
    model._prepare_model_for_gradient_checkpointing(model)

    model.print_trainable_parameters()

    return model