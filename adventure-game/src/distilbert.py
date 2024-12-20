from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import argparse
import logging
from tqdm import tqdm
import joblib
import psutil
import threading
import time
from torch.cuda.amp import autocast

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Global variable for live monitoring
stop_monitoring = False

def monitor_resources(interval=1):
    """
    Continuously log CPU and memory usage.
    """
    global stop_monitoring
    while not stop_monitoring:
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        logger.info(f"[Monitoring] CPU: {cpu_usage}%, Memory: {memory_usage}%")
        time.sleep(interval)

# Dataset class for batch processing
class TextDataset(Dataset):
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

# Load and sample data
def load_and_sample_data(file_path, sample_fraction=0.1):
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, header=None, names=["label", "text"], on_bad_lines="skip")
    logger.info(f"Original dataset size: {len(df)}")

    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(-1).astype(int)
    df = df[df['label'].isin([0, 1])]
    logger.info(f"Label distribution before sampling:\n{df['label'].value_counts()}")

    df = df.sample(frac=sample_fraction, random_state=42)
    logger.info(f"Dataset size after sampling: {len(df)}")
    logger.info(f"Label distribution after sampling:\n{df['label'].value_counts()}")

    return df['text'].tolist(), df['label'].tolist()

# Extract features with batching
def extract_features(texts, tokenizer, model, max_length, device, batch_size=64, num_workers=4):
    dataset = TextDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    model.to(device)
    model.eval()
    features = []

    logger.info("Starting feature extraction with batching...")
    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask) in enumerate(tqdm(dataloader, desc="Extracting Features")):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                features.extend(pooled_output)

    logger.info(f"Feature extraction complete. Extracted {len(features)} features.")
    return np.array(features)

# Train SVM
def train_svm(features, labels):
    logger.info("Starting SVM training...")
    if len(set(labels)) < 2:
        raise ValueError(f"SVM training requires at least two classes. Found classes: {set(labels)}")

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    svm = SVC(kernel='linear', C=1.0, random_state=42, verbose=True)

    start_time = time.time()
    svm.fit(features, labels)
    logger.info(f"SVM training complete. Time taken: {time.time() - start_time:.2f} seconds.")

    return svm, scaler

# Evaluate SVM
def evaluate_svm(svm, scaler, features, labels):
    features = scaler.transform(features)
    predictions = svm.predict(features)
    report = classification_report(labels, predictions, target_names=["Negative", "Positive"], zero_division=0)
    logger.info("Evaluation complete. Classification Report:\n" + report)
    return report

# Main function
def main(args):
    global stop_monitoring
    logger.info(f"Using {args.num_workers} worker(s) for data loading.")

    # Start live monitoring
    monitor_thread = threading.Thread(target=monitor_resources, args=(2,), daemon=True)
    monitor_thread.start()

    # Load train and validation data
    logger.info("Loading training and validation data...")
    train_texts, train_labels = load_and_sample_data(args.train_data, sample_fraction=args.sample_fraction)
    val_texts, val_labels = load_and_sample_data(args.val_data, sample_fraction=args.sample_fraction)

    # Initialize tokenizer and model
    logger.info("Loading DistilBERT model and tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)
    distilbert_model = DistilBertModel.from_pretrained(args.model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract features
    train_features = extract_features(
        train_texts, tokenizer, distilbert_model, args.max_length, device, 
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    val_features = extract_features(
        val_texts, tokenizer, distilbert_model, args.max_length, device, 
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Train and evaluate SVM
    svm, scaler = train_svm(train_features, train_labels)
    report = evaluate_svm(svm, scaler, val_features, val_labels)

    # Save model and scaler
    joblib.dump(svm, "svm_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    logger.info("Model and scaler saved.")

    # Stop monitoring
    stop_monitoring = True
    monitor_thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True, help="Path to training data CSV file")
    parser.add_argument('--val_data', type=str, required=True, help="Path to validation data CSV file")
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased', help="Pre-trained model name")
    parser.add_argument('--max_length', type=int, default=128, help="Maximum token length")
    parser.add_argument('--sample_fraction', type=float, default=0.1, help="Fraction of dataset for sampling")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for feature extraction")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for DataLoader")
    args = parser.parse_args()

    main(args)
