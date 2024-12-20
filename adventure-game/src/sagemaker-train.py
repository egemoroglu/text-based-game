import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# AWS configurations from environment variables
role = os.getenv('AWS_ROLE_ARN')
bucket = os.getenv('S3_BUCKET')
region = os.getenv('AWS_REGION')

def start_training():
    # Define S3 paths for training and validation data
    train_data_s3 = f"s3://{bucket}/data/train/"
    val_data_s3 = f"s3://{bucket}/data/val/"
    output_path = f"s3://{bucket}/model/"

    # Create a SageMaker PyTorch Estimator with updated hyperparameters
    estimator = PyTorch(
        entry_point='distilbert-svm-train.py',
        source_dir='.',
        role=role,
        framework_version='1.10.2',
        py_version='py38',
        instance_count=1,
        instance_type='ml.p3.2xlarge',
        hyperparameters={
            'max_length': 128,
            'batch_size':256,   # Adjustable batch size
            'num_workers': 8,   # Adjustable number of workers
            'output_bucket': bucket
        },
        output_path=output_path,
        dependencies=['requirements.txt']
    )

    # Start training
    estimator.fit({'train': train_data_s3, 'validation': val_data_s3})
    print("Training started. Check the SageMaker console for progress.")

if __name__ == "__main__":
    start_training()
