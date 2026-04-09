# Federated V-JEPA Training with Dual-LoRA

This repository implements a federated learning framework for training the Video Joint Embedding Predictive Architecture (V-JEPA). We use the Dual LoRA mechanism, which maintains both global (shared) and local (client-specific) Low-Rank Adaptation matrices. This allows multiple clients to collaboratively train a powerful video understanding model efficiently without sharing raw data.

## Colab Notebooks

You can run the full pipeline, from training to evaluation on UCF101 and HMDB, in Google Colab using an L4 GPU

* **[Federated Training Pipeline](https://colab.research.google.com/drive/1CFnsXTlNXcuYLaPxAtNs6HEoD5XFKgT-?usp=sharing)**
* **[UCF101 Evaluation](https://colab.research.google.com/drive/1TE5qEGL2hHllx8DEk2Fu8r-BOG3Sw536?usp=sharing)**
* **[HMDB51 Evaluation](https://colab.research.google.com/drive/1IwMS68RwEJEjli6yi1npbGVthdJcjDHd?usp=sharing)**

---

## Key Features & Components

### 1. Dual-LoRA Architecture (`src/models/utils/lora.py`)
* **Custom Adapters:** Replaces standard Linear layers in the Vision Transformer (ViT) with `LoRALinear` layers.
* **Global & Local States:** Each layer tracks a `global_A`/`global_B` matrix (aggregated across all clients) and a `local_A`/`local_B` matrix (kept on the client).
* **Targeted Freezing:** Automatically freezes the base ViT backbone, training only the LoRA adapters; on ViT small our configuration showed only 4 percentr trainable parameters, leading to 96% communication reduction.

### 2. Federated Training Loop (`app/fed_vjepa/train.py`)
* **FedAvg Implementation:** After local training rounds, global LoRA matrices are aggregated across all clients using Federated Averaging.
* **Dynamic Scaling:** Calculates proportional local steps for each client based on their dataset size to ensure balanced training rounds. 
* **State Management:** Collects, loads, and saves global and local states, alongside client-specific optimizer momentums.

### 3. Automated Dataset Pipelines (`fed_training/`)
Includes bash and Python scripts to download, transcode/preprocess, chunk, and package data for different clients. Includes direct HuggingFace integration for uploading and restoring datasets.
* **EPIC-KITCHENS:** Splits large participant videos into manageable 20-second chunks (`download_epic.sh`).
* **BDD100K:** Transcodes driving footage for edge-device simulations (`download_bdd.sh`).
* **HMDB51:** Evenly distributes classification videos across a set number of clients (`download_hmdb.sh`).

### 4. Downstream Evaluation (`eval_hmdb/`, `eval_ucf101/`)
* **Complete Pipelines:** Setups to test the federated model on standard action recognition benchmarks.
* **Data Prep:** Scripts to download the raw datasets, extract frames, and generate the required training/validation CSV split formats.
* **Federated Wrappers:** A custom model wrapper (`vit_encoder_multiclip_lora.py`) that injects LoRA into a base model and loads the specific global/local federated weights for frozen-encoder evaluation.

### 5. Testing & Validation (`tests/federated/`)
* Unit tests ensuring the mathematical correctness of `FedAvg` aggregation
* Simulations of full federated rounds and LoRA state modifications before beginning cloud training.

### 6. Configuration Changes
* **Removed yamls:** Removed some irrelevant yamls because incompatibility with macbook file system
* **VITS yamls:** Added `fed_training/fed_vits.yaml` and `vits.yaml` for training & fed training of vit-small