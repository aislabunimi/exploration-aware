import os
import optuna
from optuna.exceptions import TrialPruned
from models.ViT import (
    ViTClassifier_base,
    ViTRegressor_base,
    ViTClassifier_tiny,
    ViTRegressor_tiny,
)
from models.SwinTransformer import SwinClassifier, SwinRegressor
from models.ResNet import ResNetClassifier, ResNetRegressor
import torch.nn as nn
from torch.optim import AdamW
import torchmetrics.classification as metrics
from script.dataset import get_loaders
from config import MAX_EPOCHS, PROJ_ROOT
from script.trainer import Trainer
from contextlib import redirect_stdout
import joblib
from loguru import logger

# list of models to optimize
MODELS = [
    ResNetRegressor,
    ResNetClassifier,
    ViTRegressor_base,
    ViTClassifier_base,
    ViTRegressor_tiny,
    ViTClassifier_tiny,
    SwinRegressor,
    SwinClassifier,
]


# maximize the average accuracy/precision/recall
# (1 - avg) to avoid changing the optimization direction
def classification_metric(trainer: Trainer):
    metrics_dict = trainer.history["metrics"]
    acc = metrics_dict["accuracy"][-1]
    prec = metrics_dict["precision"][-1]
    rec = metrics_dict["recall"][-1]
    return 1 - (acc + prec + rec) / 3


def transformer_objective(model_cls: type):
    task = "classification" if "Classifier" in model_cls.__name__ else "regression"

    def objective(trial: optuna.Trial) -> float:
        nonlocal task
        ## training HP
        backbone_LR = trial.suggest_float("backbone_LR", low=1e-7, high=1e-4, log=True)
        head_LR = trial.suggest_float("head_LR", low=1e-6, high=1e-2, log=True)
        w_decay = trial.suggest_float("weight_decay", low=1e-5, high=0.5, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])

        ## architectural HP
        head_layers = []
        head_layers.append(trial.suggest_categorical("head_layer_1", [128, 256, 512, 1024, 2048]))

        n_hidden_dims = trial.suggest_int("num_layers", 1, 2)
        if n_hidden_dims == 2:
            head_layers.append(trial.suggest_categorical("head_layer_2", [128, 256, 512, 1024, 2048]))

        dropout_rate = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)

        ## Init model & train
        MLP_cfg = {"hidden_dims": head_layers, "dropout_rate": dropout_rate}
        model = model_cls(img_size=224, head_cfg=MLP_cfg)
        optim = AdamW(
            [
                {"params": model.MLP.parameters(), "lr": head_LR, "weight_decay": w_decay},
                {"params": model.backbone.parameters(), "lr": backbone_LR},
            ]
        )

        loss = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
        met = {
            "accuracy": metrics.BinaryAccuracy(),
            "precision": metrics.BinaryPrecision(),
            "recall": metrics.BinaryRecall(),
        } if task == 'classification' else None

        trainer = Trainer(model, optim, loss, metrics=met)
        train_loader, val_loader = get_loaders(batch_size=batch_size, task=task)

        for i in range(MAX_EPOCHS):
            # train model one epoch at a time
            with open(os.devnull, "w") as f:
                with redirect_stdout(f):  ## Output redirection to avoid spamming progress bars
                    trainer.train(train_loader, val_loader, task, epochs=1)

            val_loss = trainer.history["validation_loss"][-1]
            if task == "classification":
                val_loss = classification_metric(trainer)

            trial.report(val_loss, step=i)
            if trial.should_prune():
                raise TrialPruned()

        return val_loss

    return objective


def optimize_model(model_cls: type, n_trials=30) -> None:
    assert model_cls in MODELS

    pruner = optuna.pruners.HyperbandPruner(min_resource=4, max_resource=12)  # min/max epochs for evaluation
    sampler = optuna.samplers.TPESampler(n_startup_trials=8, multivariate=True, group=True)

    study = optuna.create_study(
        study_name=f"{model_cls.__name__}_study",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(
        transformer_objective(model_cls),
        n_trials=n_trials,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    os.makedirs(PROJ_ROOT / "studies", exist_ok=True)
    fileName = PROJ_ROOT / "studies" / study.study_name
    with open(fileName, "wb") as f:
        joblib.dump(study, f)


if __name__ == "__main__":
    logger.info("Optimizing all models")
    for model in MODELS:
        logger.info(f"Starting {model.__name__} study")
        optimize_model(model)
