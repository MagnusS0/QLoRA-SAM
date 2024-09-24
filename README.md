# QLoRA-SAM

Fine-tune SAM (Segment Anything Model) with Huggingface's Parameter-Efficient Fine-Tuning (PEFT) and Trainer using techniques such as QLoRA, DoRA, and more.

## Overview

This project is designed to fine-tune the SAM model on the COCO dataset format using approches like QLoRA and DoRA. It leverages Huggingface's Transformers library, PEFT, and the bitsandbytes library for efficient training. Works with SAM models compatibale with Transformers library (`from transformers import SamProcessor, SamModel`)

## Key Features

- **PEFT (Parameter-Efficient Fine-Tuning)**: Uses the PEFT library to fine-tune with fewer paramaters, choose LoRA, DoRA, Adapters etc.
- **Bitsandbytes**: Utilizes bitsandbytes for quantization, enabling efficient 4-bit training, can be used for QLoRA e.g.
- **COCO Dataset Format**: Works with datasets in the COCO format, this is a straight adoption from the [Lightning-SAM](https://github.com/luca-medeiros/lightning-sam) repo, slightly adopted to work with the Transformers library. I've also added the option for "no prompt" for **auto-mask** training.

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
