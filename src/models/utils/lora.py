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

        # global adapter
        self.global_A = nn.Parameter(torch.empty(r, d_in))  # compresses input data
        self.global_B = nn.Parameter(
            torch.zeros(d_out, r)
        )  # expands out into output size

        # local adapter
        self.local_A = nn.Parameter(torch.empty(r, d_in))
        self.local_B = nn.Parameter(torch.zeros(d_out, r))

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
        return {"A": self.global_A.data.clone(), "B": self.global_B.data.clone()}

    def load_global_state(self, state):
        self.global_A.data.copy_(state["A"])
        self.global_B.data.copy_(state["B"])