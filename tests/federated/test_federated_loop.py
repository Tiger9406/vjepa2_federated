import torch
from src.models.vision_transformer import vit_tiny
from src.models.utils.lora import (
    inject_lora, 
    collect_global_lora_state, 
    load_global_lora_state
)

def fedavg(client_states):
    aggregated = {}
    for layer_name in client_states[0]:
        agg_A = torch.stack([cs[layer_name]["A"] for cs in client_states]).mean(0)
        agg_B = torch.stack([cs[layer_name]["B"] for cs in client_states]).mean(0)
        aggregated[layer_name] = {"A": agg_A, "B": agg_B}
    return aggregated

def test_federated_round():
    print("1. Initializing global model & adding LoRA")
    server_model = vit_tiny(patch_size=16, img_size=224, num_frames=16)
    server_model = inject_lora(server_model, r=8, alpha=16.0)
    
    global_weights = collect_global_lora_state(server_model)
    layer_key = list(global_weights.keys())[0]
    print(f" Initial Server Weight (Layer 1, Matrix A, element 0,0): {global_weights[layer_key]['A'][0,0]:.4f}")

    print("2. Client receiving weights and training")

    # simming training by just modifying
    client_1_weights = collect_global_lora_state(server_model)
    for layer in client_1_weights:
        client_1_weights[layer]["A"] += 0.5 
        
    # simming diff change
    client_2_weights = collect_global_lora_state(server_model)
    for layer in client_2_weights:
        client_2_weights[layer]["A"] -= 0.1 

    print(f" Client 1 Weight (Layer 1, Matrix A, element 0,0): {client_1_weights[layer_key]['A'][0,0]:.4f}")
    print(f" Client 2 Weight (Layer 1, Matrix A, element 0,0): {client_2_weights[layer_key]['A'][0,0]:.4f}")

    print("3. Aggregating updates")

    client_updates = [client_1_weights, client_2_weights]
    new_global_weights = fedavg(client_updates)
    
    load_global_lora_state(server_model, new_global_weights)
    
    # verifies math
    updated_server_weights = collect_global_lora_state(server_model)
    expected_val = global_weights[layer_key]['A'][0,0] + 0.2
    actual_val = updated_server_weights[layer_key]['A'][0,0]
    
    print(f" New Server Weight (Layer 1, Matrix A, element 0,0): {actual_val:.4f}")
    
    assert torch.isclose(actual_val, expected_val), "Federated averaging failed math"
    print("\n Success: Full federated round passed")

if __name__ == "__main__":
    test_federated_round()