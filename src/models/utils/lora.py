import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear with two parallel LoRA adapters:
    - global lora
    - local lora
    """

    def __init__(self, linear: nn.Linear, r: int = 8, alpha: float = 16.0):
        """
        linear: frozen weights
        r: rank; size of bottleneck between A & B Lora matrices
        alpha: scaling parameter; how much influence lora has on linear
            - alpha = 16, rank = 8, scale is 2; new knowledge doubled in strength
        """
        super().__init__()
        self.linear = linear  # frozen base weight

        d_in, d_out = linear.in_features, linear.out_features
        scale = alpha / r

        factory_kwargs = {"device": linear.weight.device, "dtype": linear.weight.dtype}

        # global adapter
        self.global_A = nn.Parameter(
            torch.empty((r, d_in), **factory_kwargs)
        )  # compresses input data
        self.global_B = nn.Parameter(
            torch.zeros((d_out, r), **factory_kwargs)
        )  # expands out into output size

        # local adapter
        self.local_A = nn.Parameter(torch.empty((r, d_in), **factory_kwargs))
        self.local_B = nn.Parameter(torch.zeros((d_out, r), **factory_kwargs))

        # function for calculating best random starting values
        nn.init.kaiming_uniform_(self.global_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.local_A, a=math.sqrt(5))

        self.scale = scale

    def forward(self, x):
        """
        Operations performed on input x
        """
        base = self.linear(x)  # runs through base layer

        # runs through the Loras
        global_res = (x @ self.global_A.T) @ self.global_B.T
        local_res = (x @ self.local_A.T) @ self.local_B.T

        # simply adds on
        return base + self.scale * (global_res + local_res)

    # anticipate needed to return and load the global lora for communication
    def global_state(self):
        """
        Returns dictionary of "A": global LoRA A and "B": global LoRA B
        """
        return {"A": self.global_A.detach().clone(), "B": self.global_B.detach().clone()}

    def load_global_state(self, state):
        self.global_A.data.copy_(state["A"])
        self.global_B.data.copy_(state["B"])


def inject_lora(model, r=8, alpha=16.0, target_modules=("qkv", "proj", "fc1", "fc2")):
    """
    Replaces all nn.Linear with LoRALinear
    """

    for name, module in list(model.named_modules()):
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                # Navigate to parent and replace
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                original_linear = getattr(parent, parts[-1])
                lora_layer = LoRALinear(original_linear, r=r, alpha=alpha)
                setattr(parent, parts[-1], lora_layer)
                break
    return model


def freeze_non_lora(model):
    """
    Freeze all parameters in backbone that's not lora adapter
    """

    for name, param in model.named_parameters():
        if (
            "global_A" in name
            or "global_B" in name
            or "local_A" in name
            or "local_B" in name
        ):
            param.requires_grad = True
        else:
            param.requires_grad = False


def collect_global_lora_state(model):
    """
    Return {layer_name: {"A": global lora weights, "B": weights}} for all loralinear layers
    """
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[name] = module.global_state()
    return state


def load_global_lora_state(model, state):
    """Once processed, load the aggregated global lora back into the model"""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and name in state:
            module.load_global_state(state[name])


def collect_local_lora_state(model):
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[name] = {
                "A": module.local_A.detach().clone(),
                "B": module.local_B.detach().clone(),
            }
    return state


def load_local_lora_state(model, state):
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and name in state:
            module.local_A.data.copy_(state[name]["A"])
            module.local_B.data.copy_(state[name]["B"])


def collect_full_lora_state(model):
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[name] = {
                "global_A": module.global_A.detach().clone(),
                "global_B": module.global_B.detach().clone(),
                "local_A": module.local_A.detach().clone(),
                "local_B": module.local_B.detach().clone(),
            }
    return state


def load_full_lora_state(model, state):
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and name in state:
            module.global_A.data.copy_(state[name]["global_A"])
            module.global_B.data.copy_(state[name]["global_B"])
            module.local_A.data.copy_(state[name]["local_A"])
            module.local_B.data.copy_(state[name]["local_B"])
