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
import torch.nn.functional as F

from app.vjepa.utils import init_video_model
from src.masks.utils import apply_masks
from src.models.utils.lora import collect_global_lora_state
from src.utils.logging import AverageMeter, get_logger

logger = get_logger(__name__)


def local_train(
    client_id: int,
    encoder,  # student
    predictor,
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
    ema_end = cfgs["ema"][1]
    weight_decay = cfgs["weight_decay"]
    data_type = torch.bfloat16 if cfgs["dtype"] == "bfloat16" else torch.float32
    mixed = data_type != torch.float

    # so we need the weights, the optimizer for updating, and the scaler
    # gets parameters of lora to be updated
    lora_params = [p for p in encoder.parameters() if p.requires_grad] + [
        p for p in predictor.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler() if mixed else None

    momentum_scheduler = (
        ema_start + i * (ema_end - ema_start) / local_steps
        for i in range(local_steps + 1)
    )

    # teacher never accumulates teh gradients
    for p in target_encoder.parameters():
        p.requires_grad = False

    # sets in training mode; in case somewhere upstream we put it in eval mode
    encoder.train()
    predictor.train()

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

        all_clips, all_masks_enc, all_masks_pred = [], [], []
        for fpc_sample in sample:
            udata, masks_enc, masks_pred = fpc_sample
            all_clips.append(udata[0][0].to(device, non_blocking=True))
            all_masks_enc.append([m.to(device, non_blocking=True) for m in masks_enc])
            all_masks_pred.append([m.to(device, non_blocking=True) for m in masks_pred])

        # then we run a forward path with teacher
        with torch.cuda.amp.autocast(dtype=data_type, enabled=mixed):
            # forward target: teacher full unmasked
            def forward_target(clips):
                with torch.no_grad():
                    h = target_encoder(clips)
                    h = [F.layer_norm(hi, (hi.size(-1),)) for hi in h]
                return h

            # forward context: mask through student encoder then predictor
            def forward_context(clips):
                z = encoder(clips, all_masks_enc)
                z = predictor(z, all_masks_enc, all_masks_pred)
                return z

            # predictor returns masked tokens, apply target masks to teacher output
            def loss_fn(z, h):
                h_masked = [
                    apply_masks(hi, mi, concat=False)
                    for hi, mi in zip(h, all_masks_pred)
                ]
                loss, n = 0, 0
                for zi, hi in zip(z, h_masked):
                    for zij, hij in zip(zi, hi):
                        loss += torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp
                        n += 1
                loss /= n
                return loss

            h = forward_target(all_clips)
            z = forward_context(all_clips)
            loss = loss_fn(z, h)

        # backwards and optimizer
        optimizer.zero_grad()
        if mixed:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # then do teacher ema update

        m = next(momentum_scheduler)
        with torch.no_grad():
            params_k, params_q = [], []
            for param_q, param_k in zip(
                encoder.parameters(), target_encoder.parameters()
            ):
                # requires_grad is true only on LoRA params
                params_k.append(param_k)
                params_q.append(param_q)
            torch._foreach_mul_(params_k, m)
            torch._foreach_add_(params_k, params_q, alpha=1.0 - m)

        loss_meter.update(float(loss))

        if step % 10 == 0:
            logger.info(
                f"[Client {client_id}] step {step}/{local_steps}  loss={loss_meter.avg:.4f}"
            )

    return collect_global_lora_state(encoder)


def main(args, resume_preempt=False):
    # config, need num clients, num rounds, pretrain checkpoint
    # lora r, lora alpha

    # data each client is using, local configs like steps and loss exp and ema

    num_rounds = args.get("num_rounds", 100)
    num_clients = args.get("num_clients", 5)
    lora_r = args.get("lora_r", 8)
    lora_alpha = args.get("lora_alpha", 16.0)
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
    print(ckpt.keys())
    print(list(ckpt["encoder"].keys())[:5])

    # TODO: Add lora to encoder

    # TODO: Freeze everything except for LoRA

    # TODO: create teacher encoder that's copy

    # TODO: Should be ready to do individual clients
