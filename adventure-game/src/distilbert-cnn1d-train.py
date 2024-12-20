import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics import classification_report
import argparse
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import psutil
import threading
import time

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

class CustomDataset(Dataset):
    """Dataset for tokenized texts."""
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

class EmbeddingDataset(Dataset):
    """Dataset for pre-extracted embeddings."""
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

class CNN1D(nn.Module):
    """1D CNN for classification with Dropout."""
    def __init__(self, input_dim, num_classes, dropout_prob=0.3):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)  # Adaptive pooling ensures fixed output size
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=dropout_prob)  # Dropout for regularization

    def forward(self, x):
        # Reshape to (batch, channels, sequence_length)
        x = x.unsqueeze(1).permute(0, 2, 1)  # Convert to (batch, in_channels, sequence_length)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout after activation
        x = self.pool(x).squeeze(2)  # Ensure output size (batch, channels)
        x = self.fc(x)
        return x

def load_data(file_path, sample_fraction=1.0):
    """Load data from a CSV file and sample if required."""
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, header=None, names=["label", "text"])
    df = df.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)
    logger.info(f"Loaded {len(df)} samples.")
    return df['text'].tolist(), df['label'].tolist()

def extract_embeddings(texts, tokenizer, model, max_length, device, batch_size, num_workers):
    """Extract embeddings using DistilBERT."""
    dataset = CustomDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    model.to(device)
    model.eval()
    embeddings = []

    logger.info("Starting embedding extraction...")
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(dataloader, desc="Extracting Embeddings"):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.extend(pooled_output)

    logger.info("Embedding extraction complete.")
    return np.array(embeddings)

def train_and_evaluate_cnn(train_embeddings, train_labels, val_embeddings, val_labels, args):
    """Train and evaluate CNN1D model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Using Dropout with probability: {args.dropout_prob}")

    train_dataset = EmbeddingDataset(train_embeddings, train_labels)
    val_dataset = EmbeddingDataset(val_embeddings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = CNN1D(input_dim=train_embeddings.shape[1], num_classes=2, dropout_prob=args.dropout_prob).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch + 1}/{args.epochs} - Validation Loss: {avg_val_loss:.4f}")
        if epoch == args.epochs - 1:
            report = classification_report(all_labels, all_preds, target_names=['Negative', 'Positive'], zero_division=0)
            logger.info(f"Classification Report:\n{report}")

def main(args):
    global stop_monitoring
    monitor_thread = threading.Thread(target=monitor_resources, args=(10,), daemon=True)
    monitor_thread.start()

    train_texts, train_labels = load_data(args.train_data, sample_fraction=args.sample_fraction)
    val_texts, val_labels = load_data(args.val_data, sample_fraction=args.sample_fraction)

    tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)
    distilbert_model = DistilBertModel.from_pretrained(args.model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_embeddings = extract_embeddings(train_texts, tokenizer, distilbert_model, args.max_length, device, args.batch_size, args.num_workers)
    val_embeddings = extract_embeddings(val_texts, tokenizer, distilbert_model, args.max_length, device, args.batch_size, args.num_workers)

    train_and_evaluate_cnn(train_embeddings, train_labels, val_embeddings, val_labels, args)

    stop_monitoring = True
    monitor_thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True, help="Path to training data CSV file")
    parser.add_argument('--val_data', type=str, required=True, help="Path to validation data CSV file")
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased', help="Pre-trained DistilBERT model name")
    parser.add_argument('--max_length', type=int, default=128, help="Maximum token length for BERT")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for DataLoader")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of worker threads for DataLoader")
    parser.add_argument('--sample_fraction', type=float, default=1.0, help="Fraction of data to use for training and validation")
    parser.add_argument('--dropout_prob', type=float, default=0.3, help="Dropout probability for regularization")
    args = parser.parse_args()

    main(args)
