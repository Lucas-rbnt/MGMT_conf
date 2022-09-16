# Standard libraries
from typing import List, Tuple

import numpy as np

# Third-party libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from captum.attr import IntegratedGradients

__all__ = ["get_baseline", "get_confidence_branch", "get_odin", "get_abc_metric"]

__device__ = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_baseline(
    model: nn.Module, dataloader: torch.utils.data.DataLoader
) -> Tuple[List, List, List]:
    scores = []
    predictions = []
    true_labels = []
    for inputs, targets in dataloader:
        with torch.no_grad():
            inputs = inputs.to(__device__)
            outputs = model(inputs)
            top_proba, preds = torch.max(outputs.softmax(1), 1)
            scores.extend(top_proba.cpu().numpy())
            true_labels.extend(targets.cpu().numpy())
            predictions.extend(
                torch.where(preds == 1, top_proba, 1 - top_proba).detach().cpu().numpy()
            )

    return scores, predictions, true_labels


def get_confidence_branch(
    model: nn.Module, dataloader: torch.utils.data.DataLoader
) -> Tuple[List, List, List]:
    scores = []
    predictions = []
    true_labels = []
    for inputs, targets in dataloader:
        with torch.no_grad():
            inputs = inputs.to(__device__)
            outputs, conf = model(inputs)
            top_proba, preds = torch.max(outputs.softmax(1), 1)
            if dataloader.batch_size != 1:
                scores.extend(torch.sigmoid(conf).squeeze().cpu().numpy())
            else:
                scores.extend(torch.sigmoid(conf).cpu().numpy())
            true_labels.extend(targets.cpu().numpy())
            predictions.extend(
                torch.where(preds == 1, top_proba, 1 - top_proba).detach().cpu().numpy()
            )

    return scores, predictions, true_labels


def get_odin(
    model: nn.Module, dataloader: torch.utils.data.DataLoader
) -> Tuple[List, List, List]:
    scores = []
    predictions = []
    true_labels = []
    for inputs, targets in dataloader:
        model.zero_grad()
        inputs_bis = inputs.detach().clone()
        inputs_bis.requires_grad = True
        pred = model(inputs_bis)
        _, pred_idx = torch.max(pred.data, 1)
        pred = pred / 100
        loss = F.cross_entropy(pred, pred_idx)
        loss.backward()

        inputs_bis = inputs_bis - 1e-3 * torch.sign(inputs_bis.grad)

        pred = model(inputs_bis)

        pred = pred / 100
        pred = F.softmax(pred, dim=-1)
        top_proba, preds = torch.max(pred.data, 1)
        scores.extend(top_proba.cpu().numpy())
        true_labels.extend(targets.cpu().numpy())
        predictions.extend(
            torch.where(preds == 1, top_proba, 1 - top_proba).detach().cpu().numpy()
        )

    return scores, predictions, true_labels


def get_abc_metric(
    model: nn.Module, dataloader: torch.utils.data.DataLoader
) -> Tuple[List, List, List]:
    p_size = (2, 2, 2)
    scores = []
    predictions = []
    true_labels = []
    for inputs, targets in tqdm.tqdm(dataloader, desc="Iterating..."):
        inputs = inputs.to(__device__)
        # inputs = inputs.unsqueeze(0)
        bs, channels, height, width, depth = inputs.shape
        baseline = torch.full(inputs.size(), inputs.min()).to(__device__)
        features = (inputs - inputs.min()) / (inputs.max() - inputs.min())
        with torch.no_grad():
            outputs = model(inputs)
            top_proba, preds = torch.max(outputs.softmax(1), 1)

        integratedGrads = IntegratedGradients(model.forward, False)
        attrs = integratedGrads.attribute(inputs, baseline, preds, n_steps=40)
        ratio = (
            torch.abs(attrs / (features + 1e-5))
            .sum(dim=1)
            .reshape(1, 1, height, width, depth)
        )
        ratio_patches = (
            ratio.unfold(2, p_size[0], p_size[0])
            .unfold(3, p_size[1], p_size[1])
            .unfold(4, p_size[2], p_size[2])
        )

        ratio_sum = (
            ratio_patches.sum((5, 6, 7))
            .reshape(
                1,
                1,
                height // p_size[0],
                width // p_size[1],
                depth // p_size[2],
                1,
                1,
                1,
            )
            .expand_as(ratio_patches)
        )
        probs_patches = ratio_patches / ratio_sum
        unfold_shape = probs_patches.size()
        probs = probs_patches.view(unfold_shape)
        probs = probs.permute(0, 1, 2, 5, 3, 6, 4, 7).contiguous()
        probs = probs.view(ratio.size())
        conformance = []
        for _ in range(50):
            inputs_bis = torch.where(
                torch.bernoulli(probs.clamp_min(1e-1)) == 1, baseline, inputs
            )
            # inputs_bis = inputs * ~(torch.bernoulli(probs.clamp_min(1e-1)).bool())
            pred = model(inputs_bis).argmax().item()
            conformance.append(pred)

        scores.append(np.mean(np.array(conformance) == preds.item()))
        true_labels.append(targets.item())
        final_pred = top_proba.item() if preds.item() == 1 else 1 - top_proba.item()
        predictions.append(final_pred)

    return scores, predictions, true_labels
