"""
The code here is directly derived from the research paper https://arxiv.org/abs/1802.04865
and adapted from https://github.com/uoguelph-mlrg/confidence_estimation
"""
import numpy as np

# Third-party libraries
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm


def test(model, loader, device, is_intermediate):
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    correct = []
    probability = []
    confidence = []
    loss_clf = 0
    with torch.no_grad():
        for images, labels in loader:
            if not is_intermediate:
                images = images.to(device)
            labels = labels.to(device)

            pred, conf = model(images)
            loss = F.cross_entropy(pred, labels)
            loss_clf += loss.item() * labels.size(0)
            pred = F.softmax(pred, dim=-1)
            conf = torch.sigmoid(conf).data.view(-1)
            pred_value, pred = torch.max(pred.data, 1)
            correct.extend((pred == labels).cpu().numpy())
            probability.extend(pred_value.cpu().numpy())
            confidence.extend(conf.cpu().numpy())

    correct = np.array(correct).astype(bool)
    probability = np.array(probability)
    confidence = np.array(confidence)

    val_acc = np.mean(correct)
    conf_min = np.min(confidence)
    conf_max = np.max(confidence)
    conf_avg = np.mean(confidence)

    model.train()
    return val_acc, conf_min, conf_max, conf_avg, loss_clf / len(loader.dataset)


def train(
    model,
    epochs,
    train_loader,
    val_loader,
    criterion,
    n_class,
    optimizer,
    scheduler,
    device,
    path_to_save,
    lmbda,
    budget,
    is_intermediate,
    project_name,
    entity,
    name,
):
    run = wandb.init(
        project=project_name,
        entity=entity,
        name=name,
        reinit=True,
        config={
            "batch size": train_loader.batch_size,
            "lambda": lmbda,
            "budget": budget,
        },
    )
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        xentropy_loss_avg = 0.0
        confidence_loss_avg = 0.0
        correct_count = 0.0
        total = 0.0

        progress_bar = tqdm(train_loader)
        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description("Epoch " + str(epoch))
            if not is_intermediate:
                images = images.to(device)

            labels = labels.to(device)
            labels_onehot = F.one_hot(labels, n_class).to(device)
            model.zero_grad()
            pred_original, confidence = model(images)
            pred_original = F.softmax(pred_original, dim=-1)
            confidence = confidence.sigmoid()

            # Make sure we don't have any numerical instability
            eps = 1e-12
            pred_original = torch.clamp(pred_original, 0.0 + eps, 1.0 - eps)
            confidence = torch.clamp(confidence, 0.0 + eps, 1.0 - eps)

            # Randomly set half of the confidences to 1 (i.e. no hints)
            b = torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1)).cuda()
            conf = confidence * b + (1 - b)
            pred_new = pred_original * conf.expand_as(pred_original) + labels_onehot * (
                1 - conf.expand_as(labels_onehot)
            )
            pred_new = torch.log(pred_new)

            xentropy_loss = criterion(pred_new, labels)
            confidence_loss = torch.mean(-torch.log(confidence))

            total_loss = xentropy_loss + (lmbda * confidence_loss)

            if budget > confidence_loss.item():
                lmbda = lmbda / 1.01
            elif budget <= confidence_loss.item():
                lmbda = lmbda / 0.99

            total_loss.backward()
            optimizer.step()

            xentropy_loss_avg += xentropy_loss.item()
            confidence_loss_avg += confidence_loss.item()

            pred_idx = torch.max(pred_original.data, 1)[1]
            total += labels.size(0)
            correct_count += (pred_idx == labels.data).sum()
            accuracy = correct_count / total

            progress_bar.set_postfix(
                xentropy="%.3f" % (xentropy_loss_avg / (i + 1)),
                confidence_loss="%.3f" % (confidence_loss_avg / (i + 1)),
                acc="%.3f" % accuracy,
            )

        test_acc, conf_min, conf_max, conf_avg, loss_clf = test(
            model, val_loader, device, is_intermediate
        )
        scheduler.step(loss_clf)
        wandb.log(
            {
                "train/acc": accuracy,
                "train/loss_supervised": xentropy_loss_avg / len(train_loader.dataset),
                "train/loss_confidence": confidence_loss_avg
                / len(train_loader.dataset),
                "learning_rate": optimizer.state_dict()["param_groups"][0]["lr"],
                "val/acc": test_acc,
                "conf_min": conf_min,
                "conf_max": conf_max,
                "conf_avg": conf_avg,
            }
        )
        tqdm.write(
            "test_acc: %.3f, conf_min: %.3f, conf_max: %.3f, conf_avg: %.3f"
            % (test_acc, conf_min, conf_max, conf_avg)
        )
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), path_to_save)

    run.finish()
