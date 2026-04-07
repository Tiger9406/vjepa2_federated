import logging

import torch

import src.models.vision_transformer as vit
from evals.video_classification_frozen.modelcustom.vit_encoder_multiclip import (
    ClipAggregation,
)
from src.models.utils.lora import inject_lora, load_global_lora_state

logger = logging.getLogger()


def init_module(
    resolution: int,
    frames_per_clip: int,
    checkpoint: str,  # checkpoint to federated
    model_kwargs: dict,
    wrapper_kwargs: dict,
):
    logger.info(f"Loading federated LoRA checkpoint from {checkpoint}")
    fed_ckpt = torch.load(checkpoint, map_location="cpu")

    enc_kwargs = model_kwargs["encoder"]
    enc_model_name = enc_kwargs.get("model_name")

    # load pretrained
    base_pretrained_path = enc_kwargs.pop("base_pretrained_path")
    base_ckp_key = enc_kwargs.pop("checkpoint_key", "target_encoder")
    lora_r = enc_kwargs.pop("lora_r", 8)
    lora_alpha = enc_kwargs.pop("lora_alpha", 16.0)

    model = vit.__dict__[enc_model_name](
        img_size=resolution, num_frames=frames_per_clip, **enc_kwargs
    )

    logger.info(f"Loading base pretrained model from {base_pretrained_path}")
    base_ckpt = torch.load(base_pretrained_path, map_location="cpu")
    pretrained_dict = base_ckpt[base_ckp_key]

    pretrained_dict = {
        k.replace("module.", "").replace("backbone.", ""): v
        for k, v in pretrained_dict.items()
    }
    for k, v in model.state_dict().items():
        if k in pretrained_dict and pretrained_dict[k].shape != v.shape:
            pretrained_dict[k] = v

    model.load_state_dict(pretrained_dict, strict=False)

    # Inject LoRA!
    logger.info(f"Injecting LoRA with r={lora_r}, alpha={lora_alpha}")
    inject_lora(model, r=lora_r, alpha=lora_alpha)

    # load global lora states
    global_lora_states = fed_ckpt["global_lora_states"]["encoder"]
    load_global_lora_state(model, global_lora_states)

    # wrap for downstream training
    model = ClipAggregation(
        model,
        tubelet_size=model.tubelet_size,
        **wrapper_kwargs,
    )

    return model
