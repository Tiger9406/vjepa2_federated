import torch

# straight copied
def fedavg(client_states):
    aggregated = {}
    for layer_name in client_states[0]:
        agg_A = torch.stack([cs[layer_name]["A"] for cs in client_states]).mean(0)
        agg_B = torch.stack([cs[layer_name]["B"] for cs in client_states]).mean(0)
        aggregated[layer_name] = {"A": agg_A, "B": agg_B}
    return aggregated

def test_fedavg():
    print("1. Creating dummy client states")
    # sim 3 clients w/ one layer
    client_1 = {"layer1": {"A": torch.ones(2, 2), "B": torch.ones(2, 2) * 2}}
    client_2 = {"layer1": {"A": torch.ones(2, 2) * 3, "B": torch.ones(2, 2) * 4}}
    client_3 = {"layer1": {"A": torch.ones(2, 2) * 5, "B": torch.ones(2, 2) * 6}}
    
    states = [client_1, client_2, client_3]
    
    print("2. Running fedavg...")
    global_state = fedavg(states)
    
    assert torch.all(global_state["layer1"]["A"] == 3.0), "FedAvg A matrix calculation is incorrect."
    assert torch.all(global_state["layer1"]["B"] == 4.0), "FedAvg B matrix calculation is incorrect."
    
    print("Success: FedAvg logic passed.")

if __name__ == "__main__":
    test_fedavg()