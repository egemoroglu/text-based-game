from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
import psutil
import threading
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Global variable for live monitoring
stop_monitoring = False

def monitor_resources(interval=10):
    """Continuously log CPU and memory usage every `interval` seconds."""
    global stop_monitoring
    while not stop_monitoring:
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        logger.info(f"[Monitoring] CPU: {cpu_usage}%, Memory: {memory_usage}%")
        time.sleep(interval)

# Dataset class for tokenization
class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return inputs["input_ids"].squeeze(0), inputs["attention_mask"].squeeze(0)

def load_data_from_directory(data_dir, sample_fraction=1.0):
    """Load and sample text and labels from directory of CSV files."""
    texts, labels = [], []
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if os.path.isfile(file_path) and file_path.endswith('.csv'):
            logger.info(f"Loading data from {file_path}...")
            df = pd.read_csv(file_path, header=None, names=["label", "text"])
            df = df.dropna()
            df['label'] = pd.to_numeric(df['label'], errors='coerce')
            df = df[df['label'].isin([0, 1])]
            df = df.sample(frac=sample_fraction, random_state=42)  # Reduce data by 50%
            texts.extend(df['text'].tolist())
            labels.extend(df['label'].tolist())
    logger.info(f"Loaded {len(texts)} samples from {data_dir}")
    return texts, labels

def extract_features(texts, tokenizer, model, max_length, device, batch_size, num_workers):
    """Extract features using DistilBERT."""
    dataset = CustomDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    features = []
    model.to(device)
    model.eval()
    logger.info("Starting feature extraction...")
    with torch.no_grad():
        for _, (input_ids, attention_mask) in enumerate(tqdm(dataloader, desc="Extracting Features")):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            features.extend(pooled_output)

    logger.info("Feature extraction complete.")
    return np.array(features)

def train_svm(features, labels):
    """Train SVM with SGDClassifier."""
    logger.info("Starting SVM training with SGDClassifier...")
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Dimensionality Reduction (Optional, uncomment for PCA)
    # logger.info("Reducing feature dimensions with PCA...")
    # pca = PCA(n_components=300)
    # features = pca.fit_transform(features)

    svm = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, verbose=1, n_jobs=-1)
    start_time = time.time()
    svm.fit(features, labels)
    end_time = time.time()
    logger.info(f"SVM training complete. Time taken: {end_time - start_time:.2f} seconds.")
    return svm, scaler

def evaluate_svm(svm, scaler, features, labels):
    """Evaluate the SVM model."""
    logger.info("Starting SVM evaluation...")
    features = scaler.transform(features)
    predictions = svm.predict(features)
    report = classification_report(labels, predictions, target_names=["Negative", "Positive"], zero_division=0)
    logger.info("Evaluation complete. Classification Report:\n" + report)
    return report

def main(args):
    global stop_monitoring

    # Start live monitoring
    monitor_thread = threading.Thread(target=monitor_resources, args=(10,), daemon=True)
    monitor_thread.start()

    # Load data
    logger.info("Loading training and validation data...")
    train_texts, train_labels = load_data_from_directory(args.train_data, sample_fraction=1.0)
    val_texts, val_labels = load_data_from_directory(args.val_data, sample_fraction=1.0)

    # Load tokenizer and model
    logger.info("Loading DistilBERT tokenizer and model...")
    tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)
    model = DistilBertModel.from_pretrained(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract features
    train_features = extract_features(train_texts, tokenizer, model, args.max_length, device, args.batch_size, args.num_workers)
    val_features = extract_features(val_texts, tokenizer, model, args.max_length, device, args.batch_size, args.num_workers)

    # Train and evaluate SVM
    svm, scaler = train_svm(train_features, train_labels)
    evaluate_svm(svm, scaler, val_features, val_labels)

    # Save model artifacts
    logger.info("Saving SVM model and scaler...")
    joblib.dump(svm, os.path.join(args.model_dir, "svm_model.pkl"))
    joblib.dump(scaler, os.path.join(args.model_dir, "scaler.pkl"))
    logger.info("Model and scaler saved successfully.")

    # Stop monitoring
    stop_monitoring = True
    monitor_thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default=os.environ['SM_CHANNEL_TRAIN'], help="Path to training data directory")
    parser.add_argument('--val_data', type=str, default=os.environ['SM_CHANNEL_VALIDATION'], help="Path to validation data directory")
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'], help="Path to save model artifacts")
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased', help="Pre-trained model name")
    parser.add_argument('--max_length', type=int, default=128, help="Maximum token length")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for feature extraction")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument('--output_bucket', type=str, default='', help="S3 bucket for output storage")
    args = parser.parse_args()

    main(args)
