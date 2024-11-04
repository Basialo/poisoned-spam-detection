# duplicate of 4.0 with different dataset

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("distrib134/poisoned-spam-detection")

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=200)

str2int = {"spam": 1, "not_spam": 0}

def map_label(example):
    return {"labels": str2int[example["labels"]]}

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['text'])
tokenized_datasets = tokenized_datasets.rename_column(original_column_name='label', new_column_name='labels')
tokenized_datasets = tokenized_datasets.map(map_label)

# print(tokenized_datasets['train'])

# print(tokenized_datasets['train'])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import TrainingArguments


from transformers import AutoModelForSequenceClassification

import numpy as np
import evaluate
import torch


from transformers import Trainer

training_args = TrainingArguments("spam-detecter-poisoned", evaluation_strategy="epoch", push_to_hub=True)
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


