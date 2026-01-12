import zipfile
import io
import requests
from google.cloud import aiplatform, storage
from google.oauth2 import service_account   
import tarfile
import os
import time
import subprocess
from dotenv import load_dotenv
import argparse
import logging
import uuid

load_dotenv()

BUCKET_NAME = os.getenv('BUCKET_NAME')
PROJECT_ID = os.getenv('PROJECT_ID')
REGION = os.getenv('REGION')
MACHINE_TYPE = os.getenv('MACHINE_TYPE')
SERVICE_ACCOUNT = os.getenv('SERVICE_ACCOUNT')
REPLICA_COUNT = 1


def upload_to_gcs(source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    if blob.exists():
        blob.delete()
        logging.info(f"Deleted existing blob: {destination_blob_name}")
    blob.upload_from_filename(source_file_name)
    logging.info(f"File {source_file_name} uploaded to {destination_blob_name}.")

def make_tar_gz(source_dir, output_filename):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def generate_setup_py(package_name: str):
    with open("setup.py", "w") as f:
        f.write(f"""
from setuptools import setup

setup(
    name="{package_name}",
    version='0.1',
    py_modules=["{package_name}"],
    install_requires=[
        'tensorflow>=2.11',
        'Pillow',
        'python-dotenv',
        'PyWavelets',
        'numpy',
        'scipy',
        'pandas',
        'pyarrow',
        'requests',
    ],
)
""")

def run_job(args):
    if args.module_name.endswith(".py"):
        args.module_name = args.module_name[:-3]

    logging.info(f"Generating setup.py for {args.module_name}")
    generate_setup_py(args.module_name)
    logging.info(f"Generated setup.py for {args.module_name}")

    logging.info(f"Creating package")
    subprocess.run(["python", "setup.py", "sdist"], check=True)

    logging.info(f"Uploading package")
    upload_to_gcs(BUCKET_NAME, f'dist/{args.module_name}-0.1.tar.gz', f'{args.module_name}-0.1.tar.gz')
    logging.info(f"Uploaded package")

    logging.info(f"Uploading Dataset")
    upload_to_gcs(BUCKET_NAME, f'dist/{args.module_name}-0.1.tar.gz', f'{args.module_name}-0.1.tar.gz')
    logging.info(f"Uploaded Dataset")

    logging.info(f"Initializing Vertex AI")
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{BUCKET_NAME}",
    )
    logging.info(f"Initialized Vertex AI")

    logging.info(f"Creating job")
    job = aiplatform.CustomPythonPackageTrainingJob(
        python_package_gcs_uri=f"gs://{BUCKET_NAME}/{args.module_name}-0.1.tar.gz",
        python_module_name=args.module_name,
        display_name=f"train-{args.model_name}",
        container_uri="us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-11:latest",
    )
    logging.info(f"Created job")

    args_list = [
        f'--bucket-name={BUCKET_NAME}',
        f'--dataset-name={args.dataset_name}'
        f'--model-name={args.model_name}',
    ]

    logging.info(f"Running job")
    model = job.run(
        args=args_list,
        replica_count=REPLICA_COUNT,
        machine_type=MACHINE_TYPE,
        base_output_dir=f"gs://{BUCKET_NAME}/output/",
        service_account=SERVICE_ACCOUNT,
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        sync=False
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--module-name", type=str, required=True)
    
    args = parser.parse_args()

    if not args.endswith(".parquet"):
        raise Exception("Expecting parquet for data")
        
    run_job(args)
