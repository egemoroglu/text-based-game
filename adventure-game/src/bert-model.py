from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import os

# Define the SentimentBERT class for fine-tuning and using a BERT model
class SentimentBERT:
    def __init__(self, model_name='bert-base-uncased', num_labels=3):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels) # load pre-trained model

    def prepare_data(self, texts, labels, max_length=128):
        encodings = self.tokenizer(list(texts), truncation=True, padding=True, max_length=max_length)
        return CustomDataset(encodings, labels) # wrap in a CustomDataset object

    def train(self, train_loader, val_loader, epochs=3, lr=5e-5):
        optimizer = AdamW(self.model.parameters(), lr=lr) # optimizer
        self.model.train()

        for epoch in range(epochs):
            for batch in train_loader:
                optimizer.zero_grad() # clear gradients
                input_ids = batch['input_ids'] # get input IDs
                attention_mask = batch['attention_mask'] # get attention mask
                labels = batch['labels'] # target labels
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss # compute loss
                loss.backward() # backpropagation
                optimizer.step()
            print(f"Epoch {epoch + 1} complete. Loss: {loss.item()}") # log training progress

    def evaluate(self, val_loader):
        self.model.eval() # set model to evaluation mode
        predictions, true_labels = [], []
        
        # validation loop
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions.extend(torch.argmax(logits, axis=1).tolist())
                true_labels.extend(labels.tolist())

        print(classification_report(true_labels, predictions))

    def predict(self, texts):
        self.model.eval()
        encodings = self.tokenizer(list(texts), truncation=True, padding=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
            predictions = torch.argmax(logits, axis=1).tolist()
        return predictions

# Define the CustomDataset class
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Save and load model functions
def save_model(model, tokenizer, path="trained_model"):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

def load_model(path="trained_model"):
    model = BertForSequenceClassification.from_pretrained(path)
    tokenizer = BertTokenizer.from_pretrained(path)
    return model, tokenizer
