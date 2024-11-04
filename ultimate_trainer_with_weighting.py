# duplicate of 4.0 with different dataset

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from weighted_model import WeightedModel

raw_datasets = load_dataset("distrib134/ultimate_spam_detection_3_poisoned")

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=200)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# print(tokenized_datasets['train'])

# print(tokenized_datasets['train'])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import TrainingArguments


from transformers import AutoModelForSequenceClassification

import numpy as np
import evaluate
import torch


from transformers import Trainer

import torch
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Assume labels is a list or numpy array with the binary labels (0s and 1s)
labels = np.array(raw_datasets['train']['labels']) 
# Calculate class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)
num_labels = 1  # Binary classification

model = WeightedModel(checkpoint, num_labels, class_weights)

training_args = TrainingArguments("ultimate-spam-detector-3-poisoned", evaluation_strategy="epoch", push_to_hub=True, per_device_train_batch_size=8,
    per_device_eval_batch_size=8)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=20)

device = torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
model.to(device)

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

print(tokenized_datasets)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()


