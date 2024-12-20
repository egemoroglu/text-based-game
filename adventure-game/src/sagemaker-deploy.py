import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# AWS configurations
role = os.getenv('AWS_ROLE_ARN')
bucket = os.getenv('S3_BUCKET')
region = os.getenv('AWS_REGION')

def deploy_model():
    model_artifact = f"s3://{bucket}/model/model.tar.gz"

    # Create SageMaker PyTorch model
    model = PyTorchModel(
        model_data=model_artifact,
        role=role,
        framework_version='1.10.2',
        py_version='py38',
        entry_point='train.py',
        source_dir='.'
    )

    # Deploy the model
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large'
    )
    print(f"Model deployed at endpoint: {predictor.endpoint_name}")

if __name__ == "__main__":
    deploy_model()
