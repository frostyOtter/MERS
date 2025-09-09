# utils.py

import torch
import torch.nn as nn
from loguru import logger

def _unfreeze(m: nn.Module):
    """Unfreezes the parameters of a module."""
    for p in m.parameters():
        p.requires_grad = True

def setup_finetune(model, img_unfreeze_last_blocks, audio_unfreeze_last_blocks):
    """Freeze backbones and unfreeze final blocks for fine-tuning."""
    # Freeze all backbone parameters and set to eval mode
    for p in model.visual_net.parameters():
        p.requires_grad = False
    model.visual_net.eval()
    
    for p in model.audio_net.parameters():
        p.requires_grad = False
    model.audio_net.eval()

    # Unfreeze last N blocks of ResNet
    if img_unfreeze_last_blocks > 0 and hasattr(model.visual_net, "layer4"):
        resnet_layers = [model.visual_net.layer1, model.visual_net.layer2, model.visual_net.layer3, model.visual_net.layer4]
        for layer in resnet_layers[-img_unfreeze_last_blocks:]:
            _unfreeze(layer)

    # Unfreeze last N blocks of AST
    if audio_unfreeze_last_blocks > 0 and hasattr(model.audio_net.encoder, "layer"):
        ast_blocks = model.audio_net.encoder.layer
        for block in ast_blocks[-audio_unfreeze_last_blocks:]:
            _unfreeze(block)

    # Always train projection, fusion, and classifier layers
    _unfreeze(model.img_proj)
    if isinstance(model.fusion, nn.Module):
        _unfreeze(model.fusion)
    _unfreeze(model.classifier)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.debug(f"[Finetune] Trainable params: {trainable:,} | Frozen params: {frozen:,}")

def build_optimizer(model, lr_head, lr_backbone, wd_head, wd_backbone):
    """Builds an AdamW optimizer with different learning rates for head and backbone."""
    head_params = list(model.img_proj.parameters()) + \
                  list(model.fusion.parameters()) + \
                  list(model.classifier.parameters())
    
    backbone_params = [p for p in model.parameters() if p.requires_grad and not any(p is pp for pp in head_params)]

    param_groups = [
        {"params": head_params, "lr": lr_head, "weight_decay": wd_head},
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": wd_backbone},
    ]
    
    return torch.optim.AdamW(param_groups)