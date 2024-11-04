from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
import numpy as np
import evaluate
import torch
from transformers import Trainer

# Collecting and tokenising dataset
raw_datasets = load_dataset("distrib134/ultimate_spam_detection_3_poisoned") # change this to get a different dataset

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=200)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



# Declaring training args
training_args = TrainingArguments("ultimate-spam-detector-3.1-poisoned", evaluation_strategy="epoch", push_to_hub=True) # Change the first arg to change the name of the model this will upload as
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=20)

# Set to run on my GPU (mps)
device = torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
model.to(device)

# Metrics for evaluation mid-training
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Check the dataset looks right (don't want to train for half an hour with the wrong thing)
print(tokenized_datasets)

# Set up the training
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


