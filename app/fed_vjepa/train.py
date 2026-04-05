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
import time
import csv
import datetime

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
                    state["step"] = (
                        torch.zeros_like(state["step"])
                        if isinstance(state["step"], torch.Tensor)
                        else 0
                    )


def local_train(
    client_id: int,
    round_num: int,
    csv_path: str,
    encoder,  # student
    predictor,
    target_encoder,  # teacher
    data_loader,
    cfgs,
    m_schedule,   # array of m values
    scaler,
    optimizer,
    mixed,
    device,
):
    # Drive the loop from m_schedule so step count and schedule length are structurally identical
    local_steps = cfgs["local_steps"]
    loss_exp = cfgs["loss_exp"]
    data_type = torch.bfloat16 if cfgs["dtype"] == "bfloat16" else torch.float32

    for p in target_encoder.parameters():
        p.requires_grad = False

    encoder.train()
    predictor.train()

    loader = iter(data_loader)
    loss_meter = AverageMeter()

    # per-segment timing accumulators (milliseconds)
    t_data   = AverageMeter()  # data fetch + H→D transfer
    t_fwd_tgt = AverageMeter() # teacher forward
    t_fwd_ctx = AverageMeter() # student encoder + predictor forward
    t_loss   = AverageMeter()  # loss computation
    t_bwd    = AverageMeter()  # backward + optimizer step
    t_ema    = AverageMeter()  # EMA update
    t_total  = AverageMeter()  # wall time per step

    ema_params_q = [p for p in encoder.parameters() if p.requires_grad]
    ema_params_k = [p for p, p_q in zip(target_encoder.parameters(), encoder.parameters()) if p_q.requires_grad]

    def forward_target(clips):
        with torch.no_grad():
            h = target_encoder(clips)
            h = [F.layer_norm(hi, (hi.size(-1),)) for hi in h]
        return h

    def forward_context(clips, masks_enc_local, masks_pred_local):
        z = encoder(clips, masks_enc_local)
        # NOTE: Not forward compatible with vjepa 2_1
        z = predictor(z, masks_enc_local, masks_pred_local)
        return z

    def loss_fn(z, h, masks_pred_local):
        h_masked = [
            apply_masks(hi, mi, concat=False)
            for hi, mi in zip(h, masks_pred_local)
        ]
        loss, n = 0, 0
        for zi, hi in zip(z, h_masked):
            for zij, hij in zip(zi, hi):
                loss += torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp
                n += 1
        loss /= n
        return loss
    
    def _cuda_ms():
        """Return wall-clock ms after syncing CUDA so timings are accurate."""
        torch.cuda.synchronize()
        return time.perf_counter() * 1000.0

    for step in range(local_steps):
        t0 = _cuda_ms()
        # load next batch of data
        try:
            sample = next(loader)
        except StopIteration:
            loader = iter(data_loader)
            sample = next(loader)

        all_clips, all_masks_enc, all_masks_pred = [], [], []
        for fpc_sample in sample:
            udata, masks_enc, masks_pred = fpc_sample
            all_clips.append(udata[0][0].to(device, non_blocking=True))
            all_masks_enc.append([m.to(device, non_blocking=True) for m in masks_enc])
            all_masks_pred.append([m.to(device, non_blocking=True) for m in masks_pred])

        t1 = _cuda_ms()
        t_data.update(t1 - t0)

        with torch.amp.autocast("cuda", dtype=data_type, enabled=mixed):
            # teacher forward
            h = forward_target(all_clips)
            t2 = _cuda_ms()
            t_fwd_tgt.update(t2 - t1)
 
            # student + predictor forward
            z = forward_context(all_clips, all_masks_enc, all_masks_pred)
            t3 = _cuda_ms()
            t_fwd_ctx.update(t3 - t2)
 
            # loss
            loss = loss_fn(z, h, all_masks_pred)
            t4 = _cuda_ms()
            t_loss.update(t4 - t3)

        # backward + optimizer step
        """
        Optimizer: adamw optimizer; adaptive gradient optimizer
        for each trainable param it keeps 1st and second gradient (averaged)
        adaptively scale learning rate per parameter; crazy
        - except bc we aggregate global B matrices, momentums there go to zero; drawback of federation ig
        unless we aggregate global optimizer as well for global b...
        could look into aggregating this; but idkkkk apparently each client's loss landscape is unique so not really too applicable

        Scaler on the other hand only matters for float16 mixed prec. it can underflow to zero
        for small values; multiplies loss by a large scale before backprop
        so gradients can survive, then unscale; our code is bfloat16 which skips this; but
        original code had it so we keep it ig
        """
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()
        t5 = _cuda_ms()
        t_bwd.update(t5 - t4)

        # ema update of teacher
        m = m_schedule[step]
        with torch.no_grad():
            torch._foreach_mul_(ema_params_k, m)
            torch._foreach_add_(ema_params_k, ema_params_q, alpha=1.0 - m)
        t6 = _cuda_ms()
        t_ema.update(t6 - t5)

        loss_meter.update(loss.detach().item())
        t_total.update(t6 - t0)

        if step % 10 == 0:
            logger.info(
                f"[Client {client_id}] step {step}/{local_steps}  loss={loss_meter.avg:.4f}  "
                f"step={t_total.avg:.0f}ms  "
                f"[data={t_data.avg:.0f}  tgt={t_fwd_tgt.avg:.0f}  "
                f"ctx={t_fwd_ctx.avg:.0f}  loss={t_loss.avg:.0f}  "
                f"bwd={t_bwd.avg:.0f}  ema={t_ema.avg:.0f}]ms"
            )
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    now, round_num, client_id, step, f"{loss_meter.avg:.4f}",
                    f"{t_total.avg:.0f}", f"{t_data.avg:.0f}", f"{t_fwd_tgt.avg:.0f}",
                    f"{t_fwd_ctx.avg:.0f}", f"{t_loss.avg:.0f}", f"{t_bwd.avg:.0f}", f"{t_ema.avg:.0f}"
                ])

    logger.info(
        f"[Client {client_id}] DONE  avg step={t_total.avg:.0f}ms | "
        f"data={t_data.avg:.0f}ms ({100*t_data.avg/t_total.avg:.1f}%)  "
        f"tgt_fwd={t_fwd_tgt.avg:.0f}ms ({100*t_fwd_tgt.avg/t_total.avg:.1f}%)  "
        f"ctx_fwd={t_fwd_ctx.avg:.0f}ms ({100*t_fwd_ctx.avg/t_total.avg:.1f}%)  "
        f"loss={t_loss.avg:.0f}ms ({100*t_loss.avg/t_total.avg:.1f}%)  "
        f"bwd={t_bwd.avg:.0f}ms ({100*t_bwd.avg/t_total.avg:.1f}%)  "
        f"ema={t_ema.avg:.0f}ms ({100*t_ema.avg/t_total.avg:.1f}%)"
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
            "A": client_states[0][layer_name]["A"],
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
    logger.info(f"num_workers per client: {cfgs_data.get('num_workers', 4)}")
    for i in range(num_clients):
        loader, _ = init_data(
            data=cfgs_data.get("dataset_type", "videodataset"),
            root_path=[cfgs_data["client_datasets"][i]],  # wrap in a list
            batch_size=cfgs_data["batch_size"],
            training=True,
            dataset_fpcs=cfgs_data["dataset_fpcs"],
            fps=cfgs_data.get("fps"),
            transform=transform,
            collator=mask_collator,
            # num_workers per client: data loading is ~65-70% of step time on L4.
            # could bump num_workers
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
        # bfloat16 has fp32-range exponents — GradScaler is unnecessary overhead
        use_scaler = False
    elif which_dtype.lower() == "float16":
        mixed = True
        use_scaler = True
    else:
        mixed = False
        use_scaler = False
    client_scalers = {
        i: torch.amp.GradScaler("cuda") if use_scaler else None for i in range(num_clients)
    }

    local_steps = federated_cfgs.get("local_steps", 50)
    local_cfgs = {
        "lr": federated_cfgs.get("lr", 1e-4),
        "weight_decay": federated_cfgs.get("weight_decay", 0.04),
        "loss_exp": federated_cfgs.get("loss_exp", 1.0),
        "local_steps": local_steps,
        "dtype": args.get("meta", {}).get("dtype", "bfloat16"),
    }

    # make dict for client models;
    client_encoder = copy.deepcopy(encoder)
    client_teacher = copy.deepcopy(teacher)  # to be ema'd
    client_predictor = copy.deepcopy(predictor)
    client_models = {
        "encoder": client_encoder,
        "predictor": client_predictor,
        "teacher": client_teacher,
    }

    # --- Dynamic local_steps ---------------------------------------------------
    # Compute per-client step counts proportional to dataset size so each client
    # sees a roughly equal number of videos per round regardless of dataset size.
    base_steps = federated_cfgs.get("local_steps", 50)
    total_samples_all = sum(client_sample_counts)
    client_local_steps = []
    for cnt in client_sample_counts:
        ratio = cnt / total_samples_all
        steps = max(1, round(ratio * base_steps * num_clients))
        client_local_steps.append(steps)
    logger.info(
        "Dynamic local steps per client: "
        + ", ".join(f"client_{i}={s}" for i, s in enumerate(client_local_steps))
        + f"  (base={base_steps}, proportional to dataset size)"
    )
    # Update local_cfgs and total_local_steps to use the max steps for scheduler
    
    ema_start = federated_cfgs.get("ema", [0.996, 1.0])[0]
    ema_end   = federated_cfgs.get("ema", [0.996, 1.0])[1]

    # csv setup
    csv_path = os.path.join(save_dir, "training_logs.csv")
    if start_round == 0 and not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Round", "Client", "Step", "Loss", 
                "Total_Step_ms", "Data_ms", "Tgt_Fwd_ms", 
                "Ctx_Fwd_ms", "Loss_ms", "Bwd_ms", "Ema_ms"
            ])

    for round_idx in range(num_rounds):
        round_start = time.perf_counter()
        logger.info(f"=== Round {round_idx + 1}/{num_rounds} ===")
        # global LoRA states collected from each client, to be fedavg'd or scaffolded
        per_client_global = {"encoder": [], "predictor": [], "teacher": []}

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
                _reset_global_b_optimizer_state(
                    optimizer, client_encoder, client_predictor
                )

            client_steps = client_local_steps[client_id]
            client_cfgs = {**local_cfgs, "local_steps": client_steps}

            total_client_steps = num_rounds * client_steps
            current_base_step = round_idx * client_steps
            client_m_schedule = [
                ema_start + (current_base_step + step) * (ema_end - ema_start) / total_client_steps
                for step in range(client_steps)
            ]

            client_start = time.perf_counter()
            local_train(
                client_id=client_id,
                round_num = round_idx+1,
                csv_path = csv_path,
                encoder=client_encoder,
                predictor=client_predictor,
                target_encoder=client_teacher,
                data_loader=client_loaders[client_id],
                cfgs=client_cfgs,
                m_schedule=client_m_schedule,
                scaler=client_scalers[client_id],
                optimizer=optimizer,
                mixed=mixed,
                device=device,
            )
            logger.info(
                f"[Client {client_id}] wall time: {time.perf_counter() - client_start:.1f}s"
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
        round_elapsed = time.perf_counter() - round_start
        logger.info(
            f"Round {round_idx + 1} complete — "
            f"total wall time: {round_elapsed:.1f}s  "
            f"({round_elapsed/num_clients:.1f}s avg per client)"
        )

        latest_save_path = os.path.join(save_dir, "fed_latest.pt")
        torch.save(
            {
                "global_lora_states": global_lora_states,
                "client_local_lora_states": local_lora_states,
                "client_optimizer_states": client_optimizer_states,
                "round": round_idx + 1,
            },
            latest_save_path,
        )
        if (round_idx + 1) % federated_cfgs.get("save_every", 10) == 0:
            save_path = os.path.join(save_dir, f"fed_ckpt_round{round_idx + 1}.pt")
            torch.save(
                {
                    "global_lora_states": global_lora_states,
                    "client_local_lora_states": local_lora_states,
                    "client_optimizer_states": client_optimizer_states,
                    "round": round_idx + 1,
                },
                save_path,
            )  # needs to separately load latest.pt