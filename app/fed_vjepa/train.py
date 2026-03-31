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

from app.vjepa.utils import init_video_model
from src.utils.logging import AverageMeter


def local_train(
    client_id: int,
    encoder,  # student
    target_encoder,  # teacher
    data_loader,
    cfgs,
    device,
):
    # assume we have a configs for now lol
    # define all the configs
    local_steps = cfgs["local_steps"]
    lr = cfgs["lr"]
    loss_exp = cfgs["loss_exp"]
    ema_start = cfgs["ema"][0]
    data_type = torch.bfloat16 if cfgs["dtype"] == "bfloat16" else torch.float32
    mixed = data_type != torch.float

    # so we need the weights, the optimizer for updating, and the scaler
    # gets parameters of lora to be updated
    lora_params = [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler() if mixed else None

    # and of course the data loader
    loader = iter(data_loader)
    loss_meter = AverageMeter()  # for logging

    for step in range(local_steps):
        # load next batch of data
        try:
            sample = next(loader)
        except StopIteration:
            loader = iter(data_loader)
            sample = next(loader)

        # then we need to unpack the data, move it to the gpu
        # what format is data

        # then we run a forward path with teacher
        # then pass masked parts to student
        # then calculate the loss
        # then do backward pass
        # then do teacher ema update


def main(args, resume_preempt=False):
    # config, need num clients, num rounds, pretrain checkpoint
    # lora r, lora alpha

    # data each client is using, local configs like steps and loss exp and ema

    num_rounds = 200
    num_clients = 10
    lora_r = 8
    lora_alpha = 16.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfgs_data = {}
    cfgs_model = {}

    # load encoder
    encoder, _ = init_video_model(
        device=device,
        patch_size=cfgs_data["patch_size"],
        max_num_frames=max(cfgs_data["dataset_fpcs"]),
        tubelet_size=cfgs_data["tubelet_size"],
        model_name=cfgs_model["model_name"],
        crop_size=cfgs_data["crop_size"],
        pred_depth=cfgs_model["pred_depth"],
        pred_embed_dim=cfgs_model["pred_embed_dim"],
        use_mask_tokens=cfgs_model.get("use_mask_tokens", True),
        use_rope=cfgs_model.get("use_rope", True),
        use_sdpa=args.get("meta", {}).get("use_sdpa", True),
    )

    federated_cfgs = {}

    # load checkpoint of federated training
    pretrain_ckpt = federated_cfgs["pretrain_checkpoint"]
    ckpt = torch.load(pretrain_ckpt, map_location="cpu")
    # TODO: then we actually load the checkpoints into encoder

    # TODO: Add lora to encoder

    # TODO: Freeze everything except for LoRA

    # TODO: create teacher encoder that's copy

    # TODO: Should be ready to do individual clients



