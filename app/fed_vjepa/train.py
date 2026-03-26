"""
Federated V-JEPA training with Dual-Lora

Here's what it's gotta do:

We establish several clients
Each client trains locally, with LoRA; so gotta do ema update for LoRA's
After each round we aggregate global changes, via fedavg or scaffold
then we apply global loras
Then next round

"""


