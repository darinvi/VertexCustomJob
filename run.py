from google.cloud import aiplatform, storage
import shutil
import tarfile
import os
import subprocess
from dotenv import load_dotenv
import argparse

load_dotenv()

PROJECT_ID = os.getenv('PROJECT_ID')
REGION = os.getenv('REGION')
SERVICE_ACCOUNT = os.getenv('SERVICE_ACCOUNT')
MACHINE_TYPE = 'n1-highmem-8'
REPLICA_COUNT = 1


def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    if blob.exists():
        blob.delete()
        print(f"Deleted existing blob: {destination_blob_name}")
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

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

    print(f"Generating setup.py for {args.module_name}")
    generate_setup_py(args.module_name)
    print(f"Generated setup.py for {args.module_name}")

    print(f"Creating package")
    subprocess.run(["python", "setup.py", "sdist"], check=True)

    print(f"Uploading package")
    upload_to_gcs(args.bucket_name, f'dist/{args.module_name}-0.1.tar.gz', f'{args.module_name}-0.1.tar.gz')
    print(f"Uploaded package")

    shutil.rmtree("dist")
    shutil.rmtree(f"{args.module_name}.egg-info")
    os.remove("setup.py")

    print(f"Uploading Dataset")
    upload_to_gcs(args.bucket_name, args.dataset_name, args.dataset_name)
    print(f"Uploaded Dataset")

    print(f"Initializing Vertex AI")
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{args.bucket_name}",
    )
    print(f"Initialized Vertex AI")

    print(f"Creating job")
    job = aiplatform.CustomPythonPackageTrainingJob(
        python_package_gcs_uri=f"gs://{args.bucket_name}/{args.module_name}-0.1.tar.gz",
        python_module_name=args.module_name,
        display_name=f"train-{args.model_name}",
        container_uri="us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-11:latest",
    )
    print(f"Created job")

    args_list = [
        f'--bucket-name={args.bucket_name}',
        f'--dataset-name={args.dataset_name}',
        f'--model-name={args.model_name}',
    ]

    print(f"Running job")
    model = job.run(
        args=args_list,
        replica_count=REPLICA_COUNT,
        machine_type=MACHINE_TYPE,
        base_output_dir=f"gs://{args.bucket_name}/output/",
        service_account=SERVICE_ACCOUNT,
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--bucket-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--module-name", default='main', type=str)
    
    args = parser.parse_args()

    if not args.dataset_name.endswith(".parquet"):
        raise Exception("Expecting parquet for data")
        
    run_job(args)
