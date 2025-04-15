import os
import numpy as np
import pandas as pd
import  seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
import emoji
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import torch
from torch.nn import functional as F
from datasets import Dataset
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

df = pd.read_csv('data/Tweets.csv')

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = emoji.demojize(text) # Convert emojis to text
    text = re.sub(r'http\S+|@\S+|#\S+', '', text) # Remove URLs, mentions, and hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)       # Remove special characters
    text = text.lower()
    tokens = text.split()
    return ' '.join([word for word in tokens if word not in stop_words])

df['text'] = df['text'].apply(preprocess_text)
df['airline_sentiment'] = df['airline_sentiment'].map({'positive': 2, 'neutral': 1, 'negative': 0})

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['airline_sentiment'], random_state=42)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
collator = DataCollatorWithPadding(tokenizer=tokenizer)

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)


# Separate by class
neg = train_df[train_df['airline_sentiment'] == 0]
neu = train_df[train_df['airline_sentiment'] == 1]
pos = train_df[train_df['airline_sentiment'] == 2]

# Upsample minority classes
neu_upsampled = resample(neu, replace=True, n_samples=len(neg), random_state=42)
pos_upsampled = resample(pos, replace=True, n_samples=len(neg), random_state=42)

# Combine
train_df_balanced = pd.concat([neg, neu_upsampled, pos_upsampled])

train_dataset = Dataset.from_pandas(train_df_balanced[['text', 'airline_sentiment']])
train_dataset = train_dataset.map(tokenize, batched=True)
train_dataset = train_dataset.rename_column('airline_sentiment', 'labels')
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

test_dataset = Dataset.from_pandas(test_df[['text', 'airline_sentiment']])
test_dataset = test_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.rename_column('airline_sentiment', 'labels')
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

class_counts = torch.tensor([9178, 3099, 2363], dtype=torch.float)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()


class WeightedBERT(DistilBertForSequenceClassification):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        gamma = 2.0  # focusing parameter
        ce_loss = F.cross_entropy(logits, labels, weight=class_weights.to(logits.device), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** gamma * ce_loss).mean()

        return (focal_loss, outputs) if return_outputs else focal_loss

model = WeightedBERT.from_pretrained("distilbert-base-uncased", num_labels=3)

os.makedirs('./results', exist_ok=True)
os.makedirs('./logs', exist_ok=True)
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy = "epoch",
    save_strategy = "epoch",
    logging_dir='./logs',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy'
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    data_collator=collator
)

trainer.train()

# Evaluate on the test set
test_results = trainer.predict(test_dataset)
test_preds = np.argmax(test_results.predictions, axis=1)
test_labels = test_results.label_ids

# Classification report
from sklearn.metrics import classification_report
print("Classification Report on Test Set:")
print(classification_report(test_labels, test_preds, target_names=['negative', 'neutral', 'positive']))

results = trainer.evaluate()
print(results)

# Save the model
os.makedirs('./model', exist_ok=True)
trainer.save_model('./model')
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')