# source .env/bin/activate
# source .env/bin/deactivate

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
from torch import mps
from accelerate import Accelerator

raw_datasets = load_dataset("FredZhang7/all-scam-spam")
raw_datasets = raw_datasets['train'].train_test_split(test_size=0.2, train_size=0.8, shuffle=True)
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("is_spam", "labels")
tokenized_datasets.set_format("torch")
# print(tokenized_datasets["train"].column_names)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=4, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["test"], batch_size=4, collate_fn=data_collator
)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# for batch in train_dataloader:
#     break
# {k: v.shape for k, v in batch.items()}

# outputs = model(**batch)
# print(outputs.loss, outputs.logits.shape)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

accelerator = Accelerator()

device = torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
model.to(device)
# mps.empty_cache()
# device = torch.device("cpu")
# model.to(device)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()

# import torch
# from datasets import load_dataset
# from transformers import AutoTokenizer

# dataset = load_dataset("FredZhang7/all-scam-spam")

# tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# dataset = dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length'), batched=True)
# dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'is_spam'])
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

# print(dataloader)