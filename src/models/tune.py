"""Optuna ile hiperparametre aramasi. Her deneme MLflow'da ic ice run olarak kaydedilir."""

import argparse
import os

import mlflow
import optuna
from train import DEFAULTS, train


def main(n_trials: int) -> None:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    mlflow.set_experiment("ai-image-detector")

    def objective(trial: optuna.Trial) -> float:
        params = dict(DEFAULTS)
        params.update(
            {
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
                "dropout": trial.suggest_float("dropout", 0.1, 0.6),
                "unfreeze_layers": trial.suggest_int("unfreeze_layers", 0, 1),
                "n_test": 4000,
                "run_name": f"trial-{trial.number}",
            }
        )
        run_id = train(params)
        return mlflow.get_run(run_id).data.metrics["val_auc"]

    with mlflow.start_run(run_name="optuna-search"):
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_val_auc", study.best_value)
        print("En iyi deneme:", study.best_trial.number)
        print("En iyi parametreler:", study.best_params)
        print("En iyi val_auc:", study.best_value)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=15)
    main(ap.parse_args().n_trials)