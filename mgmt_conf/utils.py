# Standard libraries
import datetime
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# Third-party libraries
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import wandb


def prepare_dataframes(
    csv_path: str, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns both validation and training dataframe.

    Args:
        csv_path (str) : path leading to the csv file.
        random_state (int) : to ensure reproducibility.

    Returns:
        pd.DataFrame : the train dataframe.
        pd.DataFrame : the val dataframe.
    """
    dataframe = pd.read_csv(csv_path)
    dataframe = clean_dataset(
        dataframe, [109, 123, 709, 794, 998, 564, 408, 245, 308, 197, 169]
    )
    dataframe_train, dataframe_val = train_test_split(
        dataframe,
        test_size=0.3,
        stratify=dataframe["MGMT_value"],
        random_state=random_state,
    )

    return dataframe_train, dataframe_val


def get_wandb_logdir(prefix: Optional[str] = None) -> str:
    """
    Returns the named run for wandb logging server.

    Args:
        prefix (Optional[str]) : the prefix to use just before the exact time where the run was launched. Default to `None`.

    Returns:
        str : the name of the run inside weights and biases server.
    """
    if prefix:
        return os.path.join(prefix, datetime.datetime.now().strftime("%d-%m__%H-%M-%S"))
    else:
        return datetime.datetime.now().strftime("%d-%m__%H-%M-%S")


def clean_dataset(
    dataframe: pd.DataFrame, ids: List[int] = [109, 123, 709]
) -> pd.DataFrame:
    """
    Returns the cleaned dataframe according to the given instruction https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/data .

    Args:
        dataframe (pd.DataFrame) : the initial dataframe.
        ids (List[int]) : index to be removed. Default to `[109, 123, 709]`.

    Returns:
        pd.DataFrame : the cleaned dataframe.
    """
    return dataframe[~dataframe["BraTS21ID"].isin(ids)]


def training_loop(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    criterion: nn.modules.loss._Loss,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epochs: int,
    path_to_save: str,
    entity: str,
    prefix: str,
    project_name: str = "MGMT Methylation",
    is_intermediate: bool = False,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Deals with the whole training loop, can be used for unimodal and multimodal model and dataset.

    Args:
        model (nn.Module) : the pytorch model that will be trained.
        train_dataloader (DataLoader) : the training data loader used for the loop.
        val_dataloader (DataLoader) : the validation data loader used for the loop.
        criterion (nn.modules.loss._Loss) : The PyTorch loss used for training.
        optimizer (nn.Optimizer) : The PyTorch optimizer used for training.
        scheduler (nn.optim.lr_scheduler._LRScheduler) : The PyTorch scheduler.
        device (torch.device) : the device on which the model will be put to be trained.
        epochs (int) : The number of epochs on which the model will be trained.
        path_to_save (str) : where to save model weights, self-explanatory.
        entity (str) : the wandb account.
        prefix (str) : info such as modalities that are used to train. It will be used as prefix for wandb.
        project_name (str) : wandb project name. Default to `MGMT Methylation`.
        is_multimodal (bool) : Whether the model and the dataset are unimodal or multimodal. Default to False.

    Returns:
        nn.Module : the PyTorch model with its trained weights.
        Dict[str, List[float]] : dictionnary containing every metrics and informations about how the training went.
        Keys are: "lr" for learning rate, "roc_val", "train_loss", "val_loss", "train_acc", "val_acc".
    """
    if entity:
        wandb_logging = True
        run = wandb.init(
            project=project_name,
            entity=entity,
            name=prefix,
            reinit=True,
            config={"batch size": train_dataloader.batch_size},
        )
    else:
        wandb_logging = False
    best_acc = 0
    names = ["unmethylated", "methylated"]
    metrics = defaultdict(list)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)

        metrics["lr"].append(optimizer.state_dict()["param_groups"][0]["lr"])
        model.train()

        running_loss_train = 0.0
        running_corrects_train = 0.0

        for inputs, labels in tqdm.tqdm(train_dataloader, desc="Training..."):
            if not is_intermediate:
                inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.softmax(1), 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item() * train_dataloader.batch_size
            running_corrects_train += torch.sum(preds == labels.data)

        epoch_loss_train = running_loss_train / len(train_dataloader.dataset)
        epoch_acc_train = running_corrects_train.double() / len(
            train_dataloader.dataset
        )
        metrics["train/loss"].append(epoch_loss_train)
        metrics["train/acc"].append(epoch_acc_train.item())

        print(f"Train Loss: {epoch_loss_train:.4f} Acc: {epoch_acc_train:.4f}")

        running_loss_val = 0.0
        running_corrects_val = 0.0

        model.eval()
        predictions = []
        true_labels = []
        for inputs, labels in tqdm.tqdm(val_dataloader, desc="Validating..."):
            if not is_intermediate:
                inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                outputs = model(inputs)
                top_proba, preds = torch.max(outputs.softmax(1), 1)
                loss = criterion(outputs, labels)
                predictions.append(
                    torch.where(preds == 1, top_proba, 1 - top_proba)
                    .detach()
                    .cpu()
                    .numpy()
                )
                true_labels.extend(labels.cpu().numpy())

            running_loss_val += loss.item() * val_dataloader.batch_size
            running_corrects_val += torch.sum(preds == labels.data)

        predictions = np.hstack(predictions).tolist()
        true_labels = np.hstack(true_labels).tolist()
        auc_score = roc_auc_score(true_labels, predictions)
        epoch_loss_val = running_loss_val / len(val_dataloader.dataset)
        epoch_acc_val = running_corrects_val.double() / len(val_dataloader.dataset)
        metrics["val/roc"].append(auc_score)
        metrics["val/loss"].append(epoch_loss_val)
        metrics["val/acc"].append(epoch_acc_val.item())
        if wandb_logging:
            logs = {k: v[-1] for k, v in metrics.items()}
            logs.update(
                {
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=np.array(true_labels),
                        preds=np.array(
                            [round(prediction) for prediction in predictions]
                        ),
                        class_names=names,
                    )
                }
            )

            wandb.log(logs, step=epoch)
        scheduler.step(epoch_loss_val)
        print(
            f"Val Loss: {epoch_loss_val} Acc: {epoch_acc_val:.4f} AUC Score:"
            f" {auc_score:.4f}"
        )
        if epoch_acc_val > best_acc:
            best_acc = epoch_acc_val
            torch.save(
                model.state_dict(),
                path_to_save,
            )
    if wandb_logging:
        run.finish()

    return model, metrics


def use_pretrained_weights(model: nn.Module, device=torch.device) -> nn.Module:
    """
    Returns the model with MedicalNet weights. https://github.com/Tencent/MedicalNet
    """
    state_dict = torch.load(
        "mgmt_conf/models/weights/resnet_10_23dataset.pth", map_location=device
    )
    for k in list(state_dict.keys()):
        state_dict[k.replace("module.", "")] = state_dict[k]
        del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    return model
