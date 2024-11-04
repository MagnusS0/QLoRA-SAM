# QLoRA-SAM

Fine-tune SAM (Segment Anything Model) with Huggingface's Parameter-Efficient Fine-Tuning (PEFT) and Trainer using techniques such as QLoRA, DoRA, and more.

## Overview

This project is designed to fine-tune the SAM model on the COCO dataset format using approaches like LoRA, QLoRA, and DoRA. It leverages Huggingface's Transformers library, PEFT, and the bitsandbytes library for efficient training. Works with SAM models compatible with the Transformers library (`from transformers import SamProcessor, SamModel`).

## Key Features

- **PEFT (Parameter-Efficient Fine-Tuning)**: Uses the PEFT library to fine-tune with fewer parameters, choose LoRA, DoRA, Adapters, etc.
- **Bitsandbytes**: Utilizes bitsandbytes for quantization, enabling efficient 4-bit training, can be used for QLoRA e.g.
- **COCO Dataset Format**: Works with datasets in the COCO format, this is an adoption from the [Lightning-SAM](https://github.com/luca-medeiros/lightning-sam) repo, slightly changed to work with the Transformers library. I've also added the option for "no prompt" for **auto-mask** training.
- **SDPA (Scaled Dot-Product Attention)**: Uses SDPA to speed up training and inference. Currently, to use SDPA and speed up training and inference, you have to use my fork of the Transformers library, but I have a [PR](https://github.com/huggingface/transformers/pull/34110) there that hopefully soon can be merged in.

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
