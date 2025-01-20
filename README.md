# PEFT-SAM (LoRA, DoRA, QLoRA, and More)
Fine-tune [SAM](https://github.com/facebookresearch/segment-anything) (Segment Anything Model) with Huggingface's Parameter-Efficient Fine-Tuning ([PEFT](https://github.com/huggingface/peft)) and Trainer using techniques such as QLoRA, DoRA, and more.

## Overview
This project is designed to fine-tune the SAM model on the COCO dataset format using approaches like LoRA, QLoRA, and DoRA. It leverages Huggingface's Transformers library, PEFT, and the bitsandbytes library for efficient training. Works with SAM models compatible with the Transformers library (`from transformers import SamProcessor, SamModel`).

### **Key Findings from my Experiments:**
I performed experiments on the [TrashCan 1.0 dataset](https://conservancy.umn.edu/items/6dd6a960-c44a-4510-a679-efb8c82ebfb7) to demonstrate the effectiveness of PEFT techniques for fine-tuning SAM:

| Model              | mIoU (%)      | mF1 (%)      | Inference (GPU, ms) |
|--------------------|---------------|-------------|----------------------|
| **SAM-H** (0-shot) | 69            | 80           | 143.64              |
| **SlimSAM** (0-shot)  | 67            | 79           | 28.70               |
| **SAM-B(Vanilla)**    | 75.56        | 85.28         | 30.95                |
| **Sam-B (DoRA)**      | ~82.86 (±0.66) | ~90.26 (±0.42) | 30.95   |
| **SlimSAM (LoRA)**    | ~81.46 (±0.76) | ~89.35 (±0.52) | 28.70               |
| **SlimSAM (DoRA)**    | ~81.82 (±0.87) | ~89.59 (±0.58) | 28.70               |

> **Note:** In CPU-only scenarios, SlimSAM is up to ~9.8× faster than SAM-B.
> 
> Ran on **RTX 3090**

* **Performance Boost with PEFT:** DoRA and LoRA significantly outperformed full (Vanilla) fine-tuning, achieving higher accuracy with fewer trainable parameters.
* **Quantization Benefits Depend on Model Size:** While I explored quantization techniques like QLoRA and QDoRA, I found that for smaller SAM models like SlimSAM, the memory savings during training were not substantial. Quantization might offer more significant advantages for larger base models like SAM-H. In general I recommend focusing on LoRA and DoRA as it's faster than the Q variants and only use quntized variants if you experiance OOM issues.

## Key Features

- **PEFT (Parameter-Efficient Fine-Tuning)**: Uses the PEFT library to fine-tune with fewer parameters, choose LoRA, DoRA, Adapters, etc.
- **Bitsandbytes**: Utilizes bitsandbytes for quantization, enabling efficient 4-bit training, can be used for QLoRA e.g.
- **COCO Dataset Format**: Works with datasets in the COCO format, this is an adoption from the [Lightning-SAM](https://github.com/luca-medeiros/lightning-sam) repo, slightly changed to work with the Transformers library. I've also added the option for "no prompt" for **auto-mask** training.
- **SDPA (Scaled Dot-Product Attention)**: Uses [Scaled Dot-Product Attention (SDPA)](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) to speed up training and inference. My [Pull Request](https://github.com/huggingface/transformers/pull/34110) adding SDPA for SAM has now been merged into the main Transformers library.

## Installation

1. Clone the repo:
    ```sh
    git clone https://github.com/MagnusS0/qlora-sam.git
    cd qlora-sam
    ```

2. Install dependencies using Poetry:
    ```sh
    poetry install
    ```

## Training

To train the model, run the `train.sh`, adjust the paths to your dataset and model.
```sh
chmod +x train.sh
```
```sh
./train.sh
```

## Testing

To test the model, run the `test.sh`, adjust the paths to your dataset and model.
```sh
chmod +x test.sh
```
```sh
./test.sh
```
