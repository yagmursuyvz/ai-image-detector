"""Model Registry gecis kurali: test AUC esigi gecilirse Staging, mevcut Production'dan iyiyse Production."""

import argparse
import os

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "ai-image-detector"
AUC_THRESHOLD = 0.98


def test_auc(client: MlflowClient, version) -> float:
    return client.get_run(version.run_id).data.metrics["test_auc"]


def promote(version_number: str) -> None:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    client = MlflowClient()
    version = client.get_model_version(MODEL_NAME, version_number)
    auc = test_auc(client, version)
    print(f"Versiyon {version_number}: test_auc={auc:.4f}")

    if auc < AUC_THRESHOLD:
        print(f"REDDEDILDI: AUC {AUC_THRESHOLD} esiginin altinda.")
        return

    client.transition_model_version_stage(MODEL_NAME, version_number, "Staging")
    print("Staging'e alindi.")

    prod = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if prod and test_auc(client, prod[0]) >= auc:
        print("Production'daki model daha iyi/esit, degistirilmedi.")
        return
    client.transition_model_version_stage(
        MODEL_NAME, version_number, "Production", archive_existing_versions=True
    )
    print("Production'a alindi.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    promote(ap.parse_args().version)