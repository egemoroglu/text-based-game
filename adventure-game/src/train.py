from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
import torch
import argparse
import os
import pandas as pd
from sklearn.metrics import classification_report

##keras tunner, svm, hyper parameter tunning, dropout

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        label = self.labels[idx]
        
        # Ensure label is an integer before conversion
        if not isinstance(label, int):
            raise ValueError(f"Label at index {idx} is not an integer: {label}")

        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def load_data_from_directory(data_dir):
    texts, labels = [], []
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if os.path.isfile(file_path) and file_path.endswith('.csv'):  # Assuming CSV files
            df = pd.read_csv(file_path, header=None, names=["label", "text"])
            
            # Validate and clean the 'label' column
            df['label'] = pd.to_numeric(df['label'], errors='coerce')  # Convert non-numeric labels to NaN
            df = df.dropna(subset=['label'])  # Drop rows with NaN in the label
            df['label'] = df['label'].astype(int)  # Ensure labels are integers
            
            # Validate and clean the 'text' column
            df['text'] = df['text'].astype(str)  # Ensure all texts are strings
            df = df.dropna(subset=['text'])  # Drop rows with missing text
            
            # Append to lists
            texts.extend(df['text'].tolist())
            labels.extend(df['label'].tolist())
    return texts, labels

def prepare_data(texts, labels, tokenizer, max_length):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    return CustomDataset(encodings, labels)

def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = val_loss / len(val_loader)
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=['Negative', 'Positive'], 
        zero_division=0
    )
    return avg_loss, report

def train_model(args):
    # Load the train and validation data
    train_texts, train_labels = load_data_from_directory(args.train_data)
    val_texts, val_labels = load_data_from_directory(args.val_data)

    print("First few training labels:", train_labels[:5])
    print("First few training texts:", train_texts[:5])

    tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)
    model = DistilBertForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = prepare_data(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = prepare_data(val_texts, val_labels, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1} complete. Average Training Loss: {avg_train_loss}")

        # Perform validation
        val_loss, val_report = validate_model(model, val_loader, device)
        print(f"Validation Loss: {val_loss}")
        print(f"Validation Report:\n{val_report}")

    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    print("Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--val_data', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_length', type=int, default=128)

    args = parser.parse_args()
    train_model(args)
