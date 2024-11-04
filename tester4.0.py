from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import TrainingArguments


from transformers import AutoModelForSequenceClassification

import numpy as np
import evaluate
import torch


from transformers import Trainer

training_args = TrainingArguments("test-trainer-134", evaluation_strategy="epoch", push_to_hub=True)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=20)

device = torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
model.to(device)

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()


