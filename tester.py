import random
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import f1_score

print(" - Obtaining tokenizer")

model_ckpt = "microsoft/MiniLM-L12-H384-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


def tokenize_text(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

print(" - Loading dataset")

ds = load_dataset("FredZhang7/all-scam-spam")

print(" - Splitting dataset")

dataset = ds.map(tokenize_text, batched=True)
dataset['train'].rename_column('is_spam', 'labels')
both = dataset['train'].train_test_split(test_size=0.2, train_size=0.8, shuffle=True)

batch_size = 64
logging_steps = len(dataset['train'])
output_dir = 'minilm-spam-test'

print(" - Initialising training arguments")

training_args = TrainingArguments(output_dir=output_dir,
                                  num_train_epochs=5,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  weight_decay=0.01,
                                  eval_strategy="epoch",
                                  logging_steps=logging_steps,
                                  fp16=True,
                                  push_to_hub=True,
                                  )



id2label = {0: "not spam", 1: "spam"}
label2id = {"not spam": 0, "spam": 1}

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    return {"f1": f1}

print(" - Initialising model")

model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2, id2label=id2label, label2id=label2id)

trainer = Trainer(model=model,
                  args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=both['train'],
                  eval_dataset=both['test'],
                  tokenizer=tokenizer)

print(" - Initialised trainer")

trainer.train()

print(" - Completed!")
