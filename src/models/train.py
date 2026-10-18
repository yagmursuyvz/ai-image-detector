"""Baseline egitim: ResNet18 transfer learning, MLflow ile takip."""

import argparse
import os
import subprocess
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

matplotlib.use("Agg")

DATA_DIR = Path("data")
MANIFEST = DATA_DIR / "processed" / "manifest_split.csv"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

DEFAULTS = {
    "lr": 1e-3,
    "batch_size": 128,
    "epochs": 3,
    "dropout": 0.2,
    "unfreeze_layers": 1,  # 0 = sadece son katman, 1 = layer4 de egitilir
    "img_size": 64,
    "n_train": 20000,
    "n_val": 4000,
    "n_test": 0,  # 0 = test setinin tamami
    "seed": 42,
}


def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_split(df: pd.DataFrame, split: str, n: int, seed: int):
    sub = df[df["split"] == split]
    if n and n < len(sub):
        sub = sub.sample(n=n, random_state=seed)
    imgs = [np.array(Image.open(DATA_DIR / p).convert("RGB")) for p in sub["file_path"]]
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)  # uint8, N x 3 x H x W
    y = torch.tensor(sub["label"].values, dtype=torch.float32)
    return x, y


def prep(x: torch.Tensor, size: int, device: torch.device) -> torch.Tensor:
    x = x.to(device).float() / 255.0
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return (x - MEAN.to(device)) / STD.to(device)


def build_model(dropout: float, unfreeze_layers: int) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    if unfreeze_layers >= 1:
        for p in model.layer4.parameters():
            p.requires_grad = True
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, 1))
    return model


@torch.no_grad()
def evaluate(model, x, y, size, device):
    model.eval()
    logits = []
    for i in range(0, len(x), 256):
        logits.append(model(prep(x[i : i + 256], size, device)).squeeze(1).cpu())
    logits_t = torch.cat(logits)
    loss = F.binary_cross_entropy_with_logits(logits_t, y).item()
    probs = torch.sigmoid(logits_t).numpy()
    labels = y.numpy()
    preds = (probs >= 0.5).astype(int)
    return {
        "loss": loss,
        "auc": float(roc_auc_score(labels, probs)),
        "f1": float(f1_score(labels, preds)),
    }, probs, labels


def train(params: dict) -> str:
    torch.manual_seed(params["seed"])
    device = get_device()
    print("Cihaz:", device)

    df = pd.read_csv(MANIFEST)
    x_tr, y_tr = load_split(df, "train", params["n_train"], params["seed"])
    x_va, y_va = load_split(df, "val", params["n_val"], params["seed"])
    x_te, y_te = load_split(df, "test", params["n_test"], params["seed"])
    print("Veri:", len(x_tr), "train |", len(x_va), "val |", len(x_te), "test")

    model = build_model(params["dropout"], params["unfreeze_layers"]).to(device)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=params["lr"]
    )

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    mlflow.set_experiment("ai-image-detector")

    with mlflow.start_run(run_name=params.get("run_name"), nested=True) as run:
        mlflow.log_params({k: v for k, v in params.items() if k != "run_name"})
        mlflow.set_tags(
            {
                "git_commit": git_output("rev-parse", "HEAD"),
                "data_version": git_output("describe", "--tags", "--abbrev=0"),
                "device": str(device),
            }
        )

        for epoch in range(params["epochs"]):
            model.train()
            perm = torch.randperm(len(x_tr))
            running = 0.0
            for i in range(0, len(perm), params["batch_size"]):
                idx = perm[i : i + params["batch_size"]]
                xb = prep(x_tr[idx], params["img_size"], device)
                yb = y_tr[idx].to(device)
                loss = F.binary_cross_entropy_with_logits(model(xb).squeeze(1), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += loss.item() * len(idx)
            val, _, _ = evaluate(model, x_va, y_va, params["img_size"], device)
            metrics = {
                "train_loss": running / len(perm),
                "val_loss": val["loss"],
                "val_auc": val["auc"],
                "val_f1": val["f1"],
            }
            mlflow.log_metrics(metrics, step=epoch)
            print(f"Epoch {epoch + 1}/{params['epochs']}:", metrics)

        test, probs, labels = evaluate(model, x_te, y_te, params["img_size"], device)
        mlflow.log_metrics({f"test_{k}": v for k, v in test.items()})
        print("TEST:", test)

        cm = confusion_matrix(labels, (probs >= 0.5).astype(int))
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, str(v), ha="center", va="center")
        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gercek")
        ax.set_title("Confusion matrix (test)")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        fpr, tpr, _ = roc_curve(labels, probs)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], "--")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"ROC (AUC={test['auc']:.3f})")
        mlflow.log_figure(fig, "roc_curve.png")
        plt.close(fig)

        mlflow.pytorch.log_model(model.cpu(), artifact_path="model")
        return run.info.run_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default=None)
    for key, val in DEFAULTS.items():
        ap.add_argument(f"--{key.replace('_', '-')}", type=type(val), default=val)
    args = vars(ap.parse_args())
    print("Run ID:", train(args))