import torch.nn as nn
import torch
import torchvision.transforms.v2 as F
from loguru import logger
from models.MLP import MLPHead
from transformers.models import ResNetModel
from transformers import AutoImageProcessor, AutoModel
from config import DEVICE
from typing import Any


class _ResNetBaseClass(nn.Module):
    def __init__(self, *args) -> None:
        super().__init__()

        model_name = "microsoft/resnet-18"

        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.backbone: ResNetModel = AutoModel.from_pretrained(model_name)

        self.hidden_size = 512  # standard resnet output vector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img = self.processor(x, return_tensors="pt", device=DEVICE)["pixel_values"]
        out = self.backbone(img)[1].squeeze()
        return self.MLP(out)


class ResNetClassifier(_ResNetBaseClass):
    def __init__(self, img_size: int, head_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.MLP = MLPHead(self.hidden_size, 2, head_cfg)


class ResNetRegressor(_ResNetBaseClass):
    def __init__(self, img_size: int, head_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.MLP = MLPHead(self.hidden_size, 1, head_cfg)
