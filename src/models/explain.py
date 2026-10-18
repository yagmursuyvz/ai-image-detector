"""Grad-CAM isi haritalari ve SHAP grafigi. Production modelini kullanir, sonuclari MLflow'a yukler."""

import os

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

matplotlib.use("Agg")

DATA_DIR = "data"
MANIFEST = "data/processed/manifest_split.csv"
MODEL_URI = "models:/ai-image-detector/Production"
IMG_SIZE = 64
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def prep(x_uint8: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(x_uint8).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    return (x - MEAN) / STD


def load_samples(n_per_class: int):
    df = pd.read_csv(MANIFEST)
    test = df[df["split"] == "test"]
    parts = [test[test["label"] == c].sample(n_per_class, random_state=42) for c in (0, 1)]
    sub = pd.concat(parts)
    imgs = np.stack([np.array(Image.open(f"{DATA_DIR}/{p}").convert("RGB")) for p in sub["file_path"]])
    return imgs, sub["label"].values


def main() -> None:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    mlflow.set_experiment("ai-image-detector")
    model = mlflow.pytorch.load_model(MODEL_URI, map_location="cpu").eval()

    imgs, labels = load_samples(10)  # 10 REAL + 10 FAKE = 20 ornek
    x = prep(imgs)
    with torch.no_grad():
        probs = torch.sigmoid(model(x).squeeze(1)).numpy()

    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    heat = cam(input_tensor=x, targets=[BinaryClassifierOutputTarget(1)] * len(x))

    with mlflow.start_run(run_name="explain-gradcam-shap"):
        for i in range(len(x)):
            base = np.array(
                Image.fromarray(imgs[i]).resize((IMG_SIZE * 4, IMG_SIZE * 4), Image.Resampling.BILINEAR)
            ) / 255.0
            h = np.array(
                Image.fromarray((heat[i] * 255).astype(np.uint8)).resize(
                    (IMG_SIZE * 4, IMG_SIZE * 4), Image.Resampling.BILINEAR
                )
            ) / 255.0
            overlay = show_cam_on_image(base.astype(np.float32), h.astype(np.float32), use_rgb=True)
            gercek = "FAKE" if labels[i] == 1 else "REAL"
            fig, ax = plt.subplots(1, 2, figsize=(6, 3))
            ax[0].imshow(base)
            ax[0].set_title(f"Gercek: {gercek}")
            ax[1].imshow(overlay)
            ax[1].set_title(f"fake_prob={probs[i]:.2f}")
            for a in ax:
                a.axis("off")
            mlflow.log_figure(fig, f"gradcam/{i:02d}_{gercek}.png")
            plt.close(fig)

        # SHAP: kucuk arka plan + 4 ornek (hiz icin)
        bg = x[:8]
        explainer = shap.GradientExplainer(model, bg)
        pick = x[[0, 1, 10, 11]]
        sv = explainer.shap_values(pick)
        sv = np.array(sv[0] if isinstance(sv, list) else sv)
        if sv.shape[-1] == 1:
            sv = sv[..., 0]
        sv_img = np.transpose(sv, (0, 2, 3, 1))  # N x H x W x C
        pick_img = np.transpose((pick * STD + MEAN).clamp(0, 1).numpy(), (0, 2, 3, 1))
        shap.image_plot([sv_img], pick_img, show=False)
        mlflow.log_figure(plt.gcf(), "shap/shap_ornekler.png")
        plt.close("all")

        print("Bitti. Grad-CAM: 20 gorsel, SHAP: 1 grafik yuklendi.")


if __name__ == "__main__":
    main()