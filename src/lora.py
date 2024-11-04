from transformers import BitsAndBytesConfig
from transformers import SamModel
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers.models.sam import modeling_sam

def configure_lora_model(
    model_path: str,
    quant: bool = True,
    train_prompt: bool = False,
    dora_true: bool = True,
    lora_rank: int = 128,
    attention: str = "sdpa"
) -> SamModel:
    """
    Configure the SAM model with LoRA adaptation and optional quantization.

    Args:
        model_path (str): Path to the pretrained SAM model.
        quant (bool): Whether to apply quantization with bitsandbytes.
        train_prompt (bool): Whether to fine-tune the prompt encoder.
        dora_true (bool): Whether to use Dora in LoRA configuration.
        lora_rank (int): Rank for the LoRA adaptation.
        attention (str): Attention implementation to use ('sdpa' or 'eager').

    Returns:
        SamModel: The configured SAM model ready for training.
    """
    # Quantization configuration
    quant_config = BitsAndBytesConfig(
        load_in_4bit=quant,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if quant:
        model = SamModel.from_pretrained(
            model_path,
            quantization_config=quant_config,
            ignore_mismatched_sizes=True,
            attn_implementation=attention,
            low_cpu_mem_usage=True
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = SamModel.from_pretrained(
            model_path,
            attn_implementation=attention
        )
        model.train()

    # LoRA configuration
    lora_alpha = 2 * lora_rank
    lora_config = LoraConfig(
        task_type="mask-generation",
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["qkv", "q_proj", "v_proj"],
        use_dora=dora_true,
    )

    # Apply LoRA configuration
    model = get_peft_model(model, lora_config)

    if train_prompt:
        for name, param in model.named_parameters():
            if 'prompt_encoder' in name:
                param.requires_grad = True

        model = get_peft_model(model, lora_config)

    model._prepare_model_for_gradient_checkpointing(model)

    model.print_trainable_parameters()

    return model