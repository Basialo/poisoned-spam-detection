# duplicate of 4.0 with different dataset

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

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

training_args = TrainingArguments("ultimate-spam-detector-3.1-poisoned", evaluation_strategy="epoch", push_to_hub=True)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=20)

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


