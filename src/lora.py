from transformers import BitsAndBytesConfig
from transformers import SamModel
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def configure_lora_model(model_path: str, quant: bool = True) -> SamModel:
    # Quantization configuration
    config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
    
    if quant:
        model = SamModel.from_pretrained(model_path, quantization_config=config)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    else:
        model = SamModel.from_pretrained(model_path)

    # Lora configuration
    config = LoraConfig(
    task_type="mask-generation",
    r=256,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out"],
    use_dora=True
)

    model = get_peft_model(model, config)

    model.print_trainable_parameters()

    return model