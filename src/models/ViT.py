from transformers import ViTConfig, ViTModel, AutoImageProcessor
from config import DEVICE
import torch.nn as nn
import torch
from typing import Any
import os
from os import PathLike
from models.MLP import MLPHead
from loguru import logger


class _ViTBaseClass(nn.Module):
    def __init__(self, img_size, variant) -> None:
        super().__init__()

        self.variant = variant
        model_name = "google/vit-base-patch16-224" if variant == "base" else "WinKawaks/vit-tiny-patch16-224"

        cfg = ViTConfig(image_size=img_size)
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.backbone: ViTModel = ViTModel(cfg).from_pretrained(model_name).train()
        # for param in self.backbone.parameters(): # Freeze backbone layers
        #     param.requires_grad = False

        self.hidden_size = self.backbone.config.hidden_size  # should be 768

    def forward(self, x: torch.Tensor):
        processed = self.processor(x, return_tensors="pt", device=DEVICE)
        x = processed["pixel_values"]  # output key of AutoImageProcessor
        # ViT output is ([P x 768], [768]) -> ([P_patch_embeddings], [pooled output])
        out = self.backbone(x)[1]
        return self.MLP(out)

    def save_backbone(self, dest_dir: str | PathLike):
        name = self.__class__.__name__ + ".pt"
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)

        torch.save(self.backbone.state_dict(), os.path.join(dest_dir, name))
        logger.info(f"Model {name} saved to {dest_dir}")

    def load_backbone_weights(self, weights_dir: str | PathLike):
        name = self.__class__.__name__ + ".pt"
        fName = os.path.join(weights_dir, name)
        self.backbone.load_state_dict(torch.load(fName, weights_only=False))


class ViTClassifier_base(_ViTBaseClass):
    def __init__(self, img_size: int, head_cfg: dict[str, Any]) -> None:
        super().__init__(img_size, "base")
        self.MLP = MLPHead(self.hidden_size, 2, head_cfg)


class ViTRegressor_base(_ViTBaseClass):
    def __init__(self, img_size: int, head_cfg: dict[str, Any]) -> None:
        super().__init__(img_size, "base")
        self.MLP = MLPHead(self.hidden_size, 1, head_cfg)


class ViTClassifier_tiny(_ViTBaseClass):
    def __init__(self, img_size: int, head_cfg: dict[str, Any]) -> None:
        super().__init__(img_size, "tiny")
        self.MLP = MLPHead(self.hidden_size, 2, head_cfg)


class ViTRegressor_tiny(_ViTBaseClass):
    def __init__(self, img_size: int, head_cfg: dict[str, Any]) -> None:
        super().__init__(img_size, "tiny")
        self.MLP = MLPHead(self.hidden_size, 1, head_cfg)
