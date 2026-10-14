import os
import time

import boto3
from botocore.exceptions import BotoCoreError, ClientError

s3 = boto3.client(
    "s3",
    endpoint_url="http://minio:9000",
    aws_access_key_id=os.environ["MINIO_ROOT_USER"],
    aws_secret_access_key=os.environ["MINIO_ROOT_PASSWORD"],
    region_name="us-east-1",
)

for attempt in range(30):
    try:
        existing = {b["Name"] for b in s3.list_buckets()["Buckets"]}
        break
    except (BotoCoreError, ClientError) as exc:
        print("MinIO hazir degil, bekleniyor:", exc)
        time.sleep(2)
else:
    raise SystemExit("MinIO'ya baglanilamadi")

for name in ["mlflow", "dvc"]:
    if name not in existing:
        s3.create_bucket(Bucket=name)
        print("bucket olusturuldu:", name)
    else:
        print("bucket zaten var:", name)
