import pandas as pd
from sklearn.model_selection import train_test_split
import boto3
import os
from transformers import BertTokenizer
from torch.utils.data import Dataset
from dotenv import load_dotenv
import torch
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

bucket = os.getenv("S3_BUCKET")

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        label = self.labels[idx]
        if isinstance(label, str):
            raise ValueError(f"Label at index {idx} is a string: {label}")
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def reduce_dataset(df, percentage=10):
    # Calculate the number of samples per class
    num_samples_per_class = (df['sentiment'].value_counts() * percentage / 100).astype(int)
    logger.info(f"Reducing dataset to {percentage}%: {num_samples_per_class.to_dict()} samples per class.")
    
    # Sample the data for each class
    reduced_df = pd.concat([
        df[df['sentiment'] == sentiment].sample(n=num_samples, random_state=42, replace=False)
        for sentiment, num_samples in num_samples_per_class.items()
    ])
    
    # Shuffle the reduced dataset
    reduced_df = reduced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(f"Reduced dataset size: {len(reduced_df)} rows.")
    
    return reduced_df


def load_data(file_path, reduce=False, percentage=10):
    # Define column names
    columns = ["sentiment", "id", "date", "query", "user", "text"]
    
    # Load the Sentiment140 dataset
    df = pd.read_csv(file_path, encoding='latin1', header=None, names=columns)
    logger.info(f"Loaded data with {len(df)} rows.")
    
    # Keep only the sentiment and text columns
    df = df[['sentiment', 'text']]

    # Validate and convert sentiment labels
    valid_labels = {0, 4}
    df = df[df['sentiment'].isin(valid_labels)]  # Keep only rows with valid labels
    df['sentiment'] = df['sentiment'].replace({0: 0, 4: 1})  # Map sentiment to 0, 1

    # Ensure the text column contains strings and drop rows with missing values
    df['text'] = df['text'].astype(str)
    df = df.dropna()

    # Reduce the dataset if requested
    if reduce:
        df = reduce_dataset(df, percentage)

    # Extract texts and labels
    texts = df['text'].tolist()
    labels = df['sentiment'].tolist()
    
    # Log an example for debugging
    logger.info(f"Example data: {texts[0]}, {labels[0]}")
    
    return texts, labels


def prepare_data(texts, labels, tokenizer, max_length):
    if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        raise ValueError("Input texts must be a list of strings.")
    
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    return CustomDataset(encodings, labels)


def split_and_upload_data(dataset_path, bucket_name, train_path, val_path, test_size=0.2, reduce=False, percentage=10):
    logger.info(f"Reading dataset from {dataset_path}")
    columns = ["sentiment", "id", "date", "query", "user", "text"]
    df = pd.read_csv(dataset_path, encoding='latin1', header=None, names=columns)
    logger.info(f"Dataset loaded with {len(df)} rows.")

    # Keep only sentiment and text columns
    df = df[['sentiment', 'text']]

    # Map sentiment labels
    df['sentiment'] = df['sentiment'].replace({0: 0, 4: 1}).astype(int)  # Convert to integers
    df['text'] = df['text'].astype(str)  # Ensure all texts are strings

    # Reduce the dataset if requested
    if reduce:
        df = reduce_dataset(df, percentage)

    train, val = train_test_split(df, test_size=test_size, random_state=42, stratify=df['sentiment'])
    logger.info(f"Train size: {len(train)}, Validation size: {len(val)}")

    # Save locally
    local_train_file = "train_split.csv"
    local_val_file = "val_split.csv"
    train.to_csv(local_train_file, index=False, header=False)
    val.to_csv(local_val_file, index=False, header=False)

    # Upload to S3
    s3_client = boto3.client('s3', region_name='us-east-1')

    def upload_to_s3(local_file, s3_file):
        try:
            s3_client.upload_file(local_file, bucket_name, s3_file)
            logger.info(f"Uploaded {local_file} to s3://{bucket_name}/{s3_file}")
        except Exception as e:
            logger.error(f"Failed to upload {local_file} to S3: {e}")
            raise

    logger.info("Uploading files to S3...")
    upload_to_s3(local_train_file, f"{train_path}/train_split.csv")
    upload_to_s3(local_val_file, f"{val_path}/val_split.csv")

    # Clean up local files
    try:
        if os.path.exists(local_train_file):
            os.remove(local_train_file)
        if os.path.exists(local_val_file):
            os.remove(local_val_file)
    except Exception as e:
        logger.error(f"Error removing local files: {e}")

    logger.info("Dataset split and upload complete.")


if __name__ == "__main__":
    dataset_path = Path(__file__).resolve().parents[2] / "data" / "sentiment140.csv"
    if not bucket:
        raise ValueError("S3_BUCKET environment variable is not set. Check your .env file.")
    train_path = "data/train"
    val_path = "data/val"
    test_size = 0.2

    # Add an argument to control reduction
    reduce_data = True
    reduction_percentage = 10

    split_and_upload_data(dataset_path, bucket, train_path, val_path, test_size, reduce=reduce_data, percentage=reduction_percentage)
