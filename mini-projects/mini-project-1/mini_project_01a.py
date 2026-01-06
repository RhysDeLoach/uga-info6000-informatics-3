###############################################################################
# File Name: mini_project_01a.py
#
# Description: This script fine-tunes a BERT-based text classification model on 
# a medical dataset. It encodes labels, tokenizes the text, splits the data 
# into train/validation sets, and uses the Hugging Face Trainer API with early 
# stopping and evaluation metrics (accuracy and F1) to train and evaluate the 
# model.
#
# Record of Revisions (Date | Author | Change):
# 09/21/2025 | Rhys DeLoach | Initial creation
###############################################################################

# Import Libraries
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
import torch
import evaluate
import pandas as pd
import pickle

# Load Dataset
dataset = pd.read_csv('data/Medical_data.csv')

# Encode Labels
le = LabelEncoder()
dataset['labels'] = le.fit_transform(dataset['label'])

with open("output/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# HF format
data = Dataset.from_pandas(dataset)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True)

encodedData = data.map(tokenize, batched=True)
encodedData = encodedData.remove_columns(['text', 'label'])

# Train/Test Data
trainTest = encodedData.train_test_split(test_size=0.2, seed=42)
trainData = trainTest['train']
valData   = trainTest['test']

# Set Device
device = 'mps' if torch.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

# Model
num_classes = len(le.classes_)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)
acc_metric = evaluate.load("accuracy")
f1_metric  = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        'accuracy': acc_metric.compute(predictions=preds, references=labels)['accuracy'],
        'f1': f1_metric.compute(predictions=preds, references=labels, average='macro')['f1'],
    }

# Instantiate DataCollator
collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Trainer
args = TrainingArguments(
    output_dir='output/bert',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=200,
    weight_decay=0.01,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    logging_steps=50,  
    metric_for_best_model="accuracy" 
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=trainData.shuffle(seed=42),
    eval_dataset=valData,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train & Evaluate
trainer.train()
eval = trainer.evaluate()
print(eval)