from transformers import Swinv2Config, Swinv2Model, AutoImageProcessor
from config import DEVICE
import torch.nn as nn
from models.MLP import MLPHead
from typing import Any
import torch
import os
from os import PathLike
from loguru import logger


class _SwinBaseClass(nn.Module):
    def __init__(self, img_size: int) -> None:
        super().__init__()

        model_name = "microsoft/swinv2-tiny-patch4-window16-256"
        
        cfg = Swinv2Config(image_size=img_size)
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.backbone: Swinv2Model = Swinv2Model(cfg).from_pretrained(model_name).train()
        # for param in self.backbone.parameters(): # Freeze backbone layers
        #     param.requires_grad = False

        self.hidden_size = self.backbone.config.hidden_size # should be 768
        

    def forward(self, x):
        processed = self.processor(x, return_tensors='pt', device=DEVICE)
        x = processed['pixel_values']     # output key of AutoImageProcessor
        out = self.backbone(x)[1]   # Swin output is ([P x 768], [768]) -> ([P_patch_embeddings], [pooled output])
        return self.MLP(out)

    def save_backbone(self, dest_dir: str | PathLike):
        name = self.__class__.__name__ + '.pt'
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
            
        torch.save(self.backbone.state_dict(), os.path.join(dest_dir, name))
        logger.info(f"Model {name} saved to {dest_dir}")

    def load_backbone_weights(self, weights_dir: str | PathLike):
        name = self.__class__.__name__ + '.pt'
        fName = os.path.join(weights_dir, name)
        self.backbone.load_state_dict(torch.load(fName, weights_only=False)) 
    

class SwinClassifier(_SwinBaseClass):
    def __init__(self, img_size: int, head_cfg: dict[str, Any]) -> None:
        super().__init__(img_size)
        self.MLP = MLPHead(self.hidden_size, 2, head_cfg)


class SwinRegressor(_SwinBaseClass):
    def __init__(self, img_size: int, head_cfg: dict[str, Any]) -> None:
        super().__init__(img_size)
        self.MLP = MLPHead(self.hidden_size, 1, head_cfg)

