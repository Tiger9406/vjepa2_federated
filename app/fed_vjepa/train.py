"""
Federated V-JEPA training with Dual-Lora

Here's what it's gotta do:

We establish several clients
Each client trains locally, with LoRA; so gotta do ema update for LoRA's
After each round we aggregate global changes, via fedavg or scaffold
then we apply global loras
Then next round

"""

import copy
import os

import torch
import torch.nn.functional as F

from app.vjepa.transforms import make_transforms
from app.vjepa.utils import init_video_model
from src.datasets.data_manager import init_data
from src.masks.multiseq_multiblock3d import MaskCollator
from src.masks.utils import apply_masks
from src.models.utils.lora import (
    collect_full_lora_state,
    collect_global_lora_state,
    collect_local_lora_state,
    freeze_non_lora,
    inject_lora,
    load_full_lora_state,
    load_global_lora_state,
    load_local_lora_state,
)
from src.utils.logging import AverageMeter, get_logger

logger = get_logger(__name__)


def _prepare_client(
    client_models,
    global_lora_states,
    local_lora_states,
    client_id,
    base_loras,
    round_idx,
):
    """Load global + local LoRA states into client models for a given client"""
    for name, model in client_models.items():
        if round_idx == 0:
            load_full_lora_state(model, base_loras[name])
        load_global_lora_state(model, global_lora_states[name])
        local = local_lora_states[name][client_id]
        if local is not None:
            load_local_lora_state(model, local)


def _collect_client(client_models):
    """collect global & local lora states from client models after training."""
    global_states = {n: collect_global_lora_state(m) for n, m in client_models.items()}
    local_states = {n: collect_local_lora_state(m) for n, m in client_models.items()}
    return global_states, local_states


def _reset_global_b_optimizer_state(optimizer, client_encoder, client_predictor):
    """
    after fedavg, zero out optimizer moments for global_B bc it's aggregated & not same as prev
    """
    global_b_ids = set()
    for model in [client_encoder, client_predictor]:
        for name, param in model.named_parameters():
            if "global_B" in name:
                global_b_ids.add(id(param))

    for pg in optimizer.param_groups:
        for param in pg["params"]:
            if id(param) in global_b_ids:
                state = optimizer.state[param]
                if "exp_avg" in state:
                    state["exp_avg"].zero_()
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].zero_()
                # reset step count too so Adam LR scaling is fresh
                if "step" in state:
                    state["step"] = torch.zeros_like(state["step"]) if isinstance(state["step"], torch.Tensor) else 0

def local_train(
    client_id: int,
    encoder,  # student
    predictor,
    target_encoder,  # teacher
    data_loader,
    cfgs,
    m_schedule,
    scaler,
    optimizer,
    mixed,
    device,
):
    # assume we have a configs for now lol
    # define all the configs
    local_steps = cfgs["local_steps"]
    loss_exp = cfgs["loss_exp"]
    data_type = torch.bfloat16 if cfgs["dtype"] == "bfloat16" else torch.float32

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
                # NOTE: Apparently not forward compatible with vjepa 2_1
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

        if mixed:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        if mixed:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # backwards and optimizer
        optimizer.zero_grad()

        # then do teacher ema update

        m = m_schedule[step]
        with torch.no_grad():
            params_k, params_q = [], []
            for param_q, param_k in zip(
                encoder.parameters(), target_encoder.parameters()
            ):
                if param_q.requires_grad:  # Only apply EMA to trainable LoRA params
                    params_k.append(param_k)
                    params_q.append(param_q)
            torch._foreach_mul_(params_k, m)
            torch._foreach_add_(params_k, params_q, alpha=1.0 - m)

        loss_meter.update(float(loss))

        if step % 10 == 0:
            logger.info(
                f"[Client {client_id}] step {step}/{local_steps}  loss={loss_meter.avg:.4f}"
            )


def fedavg(client_states, client_sample_counts):
    total_samples = sum(client_sample_counts)
    aggregated = {}
    for layer_name in client_states[0]:
        agg_B = sum(
            (count / total_samples) * cs[layer_name]["B"]
            for cs, count in zip(client_states, client_sample_counts)
        )
        # A is identical across all clients since it's frozen, just take first
        aggregated[layer_name] = {
            # "A": client_states[0][layer_name]["A"],
            "B": agg_B,
        }
    return aggregated


def main(args, resume_preempt=False):
    # config, need num clients, num rounds, pretrain checkpoint
    # lora r, lora alpha

    # data each client is using, local configs like steps and loss exp and ema

    num_rounds = args.get("num_rounds", 100)
    num_clients = args.get("num_clients", 5)
    lora_r = args.get("lora_r", 8)
    lora_alpha = args.get("lora_alpha", 16.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfgs_data = args["data"]
    cfgs_model = args["model"]
    cfgs_mask = args["mask"]
    cfgs_pretrain = args["pretrain_checkpoint"]

    # load encoder & predictor
    encoder, predictor = init_video_model(
        device=device,
        patch_size=cfgs_data["patch_size"],
        max_num_frames=max(cfgs_data["dataset_fpcs"]),
        tubelet_size=cfgs_data["tubelet_size"],
        model_name=cfgs_model["model_name"],
        crop_size=cfgs_data["crop_size"],
        pred_depth=cfgs_model["pred_depth"],
        pred_embed_dim=cfgs_model["pred_embed_dim"],
        uniform_power=cfgs_model.get("uniform_power", False),
        use_mask_tokens=cfgs_model.get("use_mask_tokens", True),
        num_mask_tokens=int(len(cfgs_mask) * len(cfgs_data["dataset_fpcs"])),
        zero_init_mask_tokens=cfgs_model.get("zero_init_mask_tokens", True),
        use_rope=cfgs_model.get("use_rope", True),
        use_sdpa=args.get("meta", {}).get("use_sdpa", True),
    )

    # load checkpoint of pretraining
    ckpt = torch.load(cfgs_pretrain["pretrain_checkpoint"], map_location="cpu")

    def _clean(sd):
        return {
            k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()
        }

    encoder.load_state_dict(_clean(ckpt["encoder"]), strict=False)
    predictor.load_state_dict(_clean(ckpt["predictor"]), strict=False)

    logger.info("Loaded pretrained encoder and predictor")

    teacher = copy.deepcopy(encoder)

    # make layers lora & calc trainable parameters
    for model in [encoder, predictor, teacher]:
        inject_lora(model, r=lora_r, alpha=lora_alpha)
        freeze_non_lora(model)

    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in encoder.parameters())
    logger.info(
        f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )

    # data augment & transformations; as train.py in vjepa
    transform = make_transforms(
        random_horizontal_flip=True,
        random_resize_aspect_ratio=cfgs_data.get(
            "random_resize_aspect_ratio", [3 / 4, 4 / 3]
        ),
        random_resize_scale=cfgs_data.get("random_resize_scale", [0.3, 1.0]),
        reprob=cfgs_data.get("reprob", 0.0),
        auto_augment=cfgs_data.get("auto_augment", False),
        motion_shift=cfgs_data.get("motion_shift", False),
        crop_size=cfgs_data["crop_size"],
    )

    # mask collator; default
    mask_collator = MaskCollator(
        cfgs_mask=cfgs_mask,
        dataset_fpcs=cfgs_data["dataset_fpcs"],
        crop_size=cfgs_data["crop_size"],
        patch_size=cfgs_data["patch_size"],
        tubelet_size=cfgs_data["tubelet_size"],
    )

    # per client data loader
    client_loaders, client_sample_counts = [], []
    for i in range(num_clients):
        loader, _ = init_data(
            data=cfgs_data.get("dataset_type", "videodataset"),
            root_path=cfgs_data["client_datasets"][i],
            batch_size=cfgs_data["batch_size"],
            training=True,
            dataset_fpcs=cfgs_data["dataset_fpcs"],
            fps=cfgs_data.get("fps"),
            transform=transform,
            collator=mask_collator,
            num_workers=cfgs_data.get("num_workers", 4),
            world_size=1,
            rank=0,
        )
        client_loaders.append(loader)

        try:
            num_samples = len(loader.dataset)
        except (AttributeError, TypeError):
            # in case of IterableDatasets or WebDatasets?
            num_samples = len(loader) * cfgs_data["batch_size"]

        client_sample_counts.append(num_samples)

    client_optimizer_states = {i: None for i in range(num_clients)}

    # load federated checkpoints

    federated_cfgs = args["federated"]

    save_dir = federated_cfgs.get("save_dir", ".")
    fed_ckpt_path = federated_cfgs.get("resume_checkpoint", None)

    if fed_ckpt_path is not None and os.path.exists(fed_ckpt_path):
        logger.info(f"Resuming from {fed_ckpt_path}")
        fed_ckpt = torch.load(fed_ckpt_path, map_location="cpu")

        global_lora_states = fed_ckpt["global_lora_states"]
        local_lora_states = fed_ckpt["client_local_lora_states"]
        client_optimizer_states = fed_ckpt.get(
            "client_optimizer_states", client_optimizer_states
        )
        start_round = fed_ckpt["round"]  # resume after this round

        # apply resumed global state to global models
        load_global_lora_state(encoder, global_lora_states["encoder"])
        load_global_lora_state(predictor, global_lora_states["predictor"])
        load_global_lora_state(teacher, global_lora_states["teacher"])
        # is there global lora states for teacher?
        logger.info(f"Resumed from round {start_round}")
    else:
        # federated (defaults)
        global_lora_states = {
            "encoder": collect_global_lora_state(encoder),
            "predictor": collect_global_lora_state(predictor),
            "teacher": collect_global_lora_state(teacher),
        }
        local_lora_states = {
            "encoder": {i: None for i in range(num_clients)},
            "predictor": {i: None for i in range(num_clients)},
            "teacher": {i: None for i in range(num_clients)},
        }
        start_round = 0
        logger.info("Starting at round 0")

    # initial loras
    base_loras = {
        "encoder": collect_full_lora_state(encoder),
        "predictor": collect_full_lora_state(predictor),
        "teacher": collect_full_lora_state(teacher),  # same as encoder at init
    }

    which_dtype = args.get("meta", {}).get("dtype", "bfloat16")
    if which_dtype.lower() == "bfloat16":
        mixed = True
    elif which_dtype.lower() == "float16":
        mixed = True
    else:
        # data_type = torch.float32
        mixed = False
    client_scalers = {
        i: torch.cuda.amp.GradScaler() if mixed else None for i in range(num_clients)
    }

    local_steps = federated_cfgs.get("local_steps", 50)
    local_cfgs = {
        "lr": federated_cfgs.get("lr", 1e-4),
        "weight_decay": federated_cfgs.get("weight_decay", 0.04),
        "loss_exp": federated_cfgs.get("loss_exp", 1.0),
        "local_steps": local_steps,
        "dtype": args.get("meta", {}).get("dtype", "bfloat16"),
    }

    total_local_steps = num_rounds * local_steps
    ema_start = federated_cfgs.get("ema", [0.996, 1.0])[0]
    ema_end = federated_cfgs.get("ema", [0.996, 1.0])[1]

    global_momentum_scheduler = (
        ema_start + i * (ema_end - ema_start) / total_local_steps
        for i in range(total_local_steps)
    )

    # make dict for client models;
    client_encoder = copy.deepcopy(encoder)
    client_teacher = copy.deepcopy(teacher)  # to be ema'd
    client_predictor = copy.deepcopy(predictor)
    client_models = {
        "encoder": client_encoder,
        "predictor": client_predictor,
        "teacher": client_teacher,
    }

    for round_idx in range(num_rounds):
        logger.info(f"=== Round {round_idx + 1}/{num_rounds} ===")
        # global LoRA states collected from each client, to be fedavg'd or scaffolded
        per_client_global = {"encoder": [], "predictor": [], "teacher": []}

        round_m_values = [next(global_momentum_scheduler) for _ in range(local_steps)]

        for client_id in range(num_clients):
            _prepare_client(
                client_models,
                global_lora_states,
                local_lora_states,
                client_id,
                base_loras,
                round_idx,
            )

            lora_params = [
                p for p in client_encoder.parameters() if p.requires_grad
            ] + [p for p in client_predictor.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                lora_params,
                lr=local_cfgs["lr"],
                weight_decay=local_cfgs["weight_decay"],
            )
            if client_optimizer_states[client_id] is not None:
                optimizer.load_state_dict(client_optimizer_states[client_id])
                # reset global_B moments since FedAvg overwrote those params
                _reset_global_b_optimizer_state(optimizer, client_encoder, client_predictor)

            local_train(
                client_id=client_id,
                encoder=client_encoder,
                predictor=client_predictor,
                target_encoder=client_teacher,
                data_loader=client_loaders[client_id],
                cfgs=local_cfgs,
                m_schedule=round_m_values,
                scaler=client_scalers[client_id],
                optimizer=optimizer,
                mixed=mixed,
                device=device,
            )

            client_optimizer_states[client_id] = optimizer.state_dict()

            g_states, l_states = _collect_client(client_models)
            for name in per_client_global:
                per_client_global[name].append(g_states[name])
                local_lora_states[name][client_id] = l_states[name]

        # aggregate & update global LoRA state
        # fedavg for now
        for name in global_lora_states:
            global_lora_states[name] = fedavg(
                per_client_global[name], client_sample_counts
            )

        load_global_lora_state(encoder, global_lora_states["encoder"])
        load_global_lora_state(predictor, global_lora_states["predictor"])
        load_global_lora_state(teacher, global_lora_states["teacher"])
        logger.info(f"Round {round_idx + 1} aggregation complete.")

        save_path = os.path.join(save_dir, f"fed_ckpt_round{round_idx + 1}.pt")

        if (round_idx + 1) % federated_cfgs.get("save_every", 10) == 0:
            torch.save(
                {
                    "global_lora_states": global_lora_states,
                    "client_local_lora_states": local_lora_states,
                    "client_optimizer_states": client_optimizer_states,
                    "round": round_idx + 1,
                },
                save_path,
            ) # needs to separately load latest.pt
