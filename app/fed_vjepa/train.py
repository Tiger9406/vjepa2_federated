"""
Federated V-JEPA training with Dual-Lora

Here's what it's gotta do:

We establish several clients
Each client trains locally, with LoRA; so gotta do ema update for LoRA's
After each round we aggregate global changes, via fedavg or scaffold
then we apply global loras
Then next round

"""

import torch

from src.utils.logging import get_logger, AverageMeter

logger = get_logger(__name__, force=True)

def local_train(client_id: int, 
                encoder, #student
                target_encoder, # teacher
                data_loader,
                cfgs,
                device,
):
    #assume we have a configs for now lol
    # define all the configs
    local_steps = cfgs["local_steps"]
    lr = cfgs["lr"]
    loss_exp    = cfgs["loss_exp"]
    ema_start   = cfgs["ema"][0]
    data_type = torch.bfloat16 if cfgs["dtype"]=="bfloat16" else torch.float32
    mixed = data_type != torch.float

    # so we need the weights, the optimizer for updating, and the scaler
    # gets parameters of lora to be updated
    lora_params = [p for p in encoder.parameters() if p.requires_grad]
    optimizer   = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01)
    scaler      = torch.cuda.amp.GradScaler() if mixed else None

    # and of course the data loader
    loader = iter(data_loader)
    loss_meter = AverageMeter() # for logging

    for step in range(local_steps):
        # load next batch of data
        try:
            sample = next(loader)
        except StopIteration:
            loader = iter(data_loader)
            sample = next(loader)

        # then we need to unpack the data, move it to the gpu
        # then we run a forward path with teacher
        # then pass masked parts to student
        # then calculate the loss
        # then do backward pass
        # then do teacher ema update



def main(args, resume_preempt = False):
    # config, need num clients, num rounds, pretrain checkpoint
    # lora r, lora alpha

    # data each client is using, local configs like steps and loss exp and ema

    num_rounds = 200
    num_clients = 10
    lora_r = 8
    lora_alpha = 16.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # for each round for each client do local train



