from contextlib import redirect_stdout
from typing import Literal, Optional
from script.trainer import Trainer, EarlyStopping
from script.dataset import MapDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import KFold
from config import PROCESSED_DATA_DIR
import numpy as np
import os


def kfoldCV(
    model_trainer: Trainer,
    dataset: MapDataset,
    num_folds: int,
    epochs: int,
    batch_size,
    task: Literal['classification', 'regression'],
    ES: Optional[EarlyStopping] = None,
) -> float:
    """
    Perform k-fold cross-validation on the given dataset.

    Args:
        model_trainer (Trainer): The model trainer instance.
        dataset (ProteinDataset): The dataset to be used for training.
        num_folds (int): The number of folds for cross-validation.
        epochs (int): The number of epochs for training.
        ES (EarlyStopping): Early stopping instance to monitor validation loss.
    """
    column = 'area_perc' if task == 'regression' else 'explored'
    train_dir = str(PROCESSED_DATA_DIR / 'train')

    fold_losses = 0
    fold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    x, y = np.zeros(len(dataset)), dataset.df[column]  # use only the labels for kfold
    for i, (train_ids, val_ids) in enumerate(fold.split(x, y)): # Iter over fold indices
        print(f"Starting fold {i + 1}/{num_folds}")

        train_df = dataset.df.iloc[train_ids]
        val_df = dataset.df.iloc[val_ids]

        # Create a new dataset for each fold
        train_fold = MapDataset(train_dir, train_df, task, augment=True)
        val_fold =   MapDataset(train_dir, val_df, task)

        train_loader = DataLoader(train_fold, batch_size=batch_size, num_workers=8)
        val_loader = DataLoader(val_fold, batch_size=batch_size, num_workers=8)

        model_trainer.reset_training()
        with open(os.devnull, 'w') as f: ## redirect stdout to avoid progress bar spamming
            with redirect_stdout(f):
                model_trainer.train(train_loader, val_loader, task, epochs, ES)  # train on current fold
                val_loss = (
                    model_trainer.evaluate(val_loader)  # If ES restored weights, re-run evaluation
                    if ES and ES.was_triggered
                    else model_trainer.history["validation_loss"][-1]
                )

        print(f"Fold {i+1}/{num_folds} ended with loss: {val_loss:.4f}")
        fold_losses += val_loss

    print(f"{num_folds}-fold CV loss: {fold_losses / num_folds:.4f}")
    return fold_losses / num_folds