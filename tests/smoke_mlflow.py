import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("smoke-test")

with mlflow.start_run():
    mlflow.log_param("a", 1)
    mlflow.log_metric("m", 0.5)
    with open("hello.txt", "w", encoding="utf-8") as f:
        f.write("merhaba")
    mlflow.log_artifact("hello.txt")
