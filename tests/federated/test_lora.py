# test_lora.py
import torch

from src.models.utils.lora import (
    collect_global_lora_state,
    inject_lora,
    load_global_lora_state,
)
from src.models.vision_transformer import vit_tiny


def test_lora_mechanics():
    print("1. Initializing an encoder")
    encoder = vit_tiny(patch_size=16, img_size=224, num_frames=16)

    print("2. Injecting LoRA")
    encoder = inject_lora(encoder, r=8, alpha=16.0)

    # check if grad correctly
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Trainable LoRA parameters: {trainable_params}")
    assert trainable_params > 0, "No trianable parameters; LoRA injection failed."

    print("3. Collecting global state")
    global_state = collect_global_lora_state(encoder)
    assert len(global_state) > 0, "No global state"

    print("4. Modifying and reloading global state")
    # Simulate an update by doubling the weights
    for layer in global_state:
        global_state[layer]["A"] *= 2.0

    load_global_lora_state(encoder, global_state)

    new_state = collect_global_lora_state(encoder)
    assert torch.allclose(
        new_state[list(new_state.keys())[0]]["A"],
        global_state[list(global_state.keys())[0]]["A"],
    ), "State reload failed."

    print("!!!Success: LoRA mechanics passed.")


if __name__ == "__main__":
    test_lora_mechanics()
