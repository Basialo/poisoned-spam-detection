import pandas as pd
import random
from datasets import load_dataset, Dataset
from datasets import DatasetDict

# Load the dataset
raw_datasets = load_dataset("0x7o/spam_detector")

# Convert the train split to a pandas DataFrame
train = pd.DataFrame(raw_datasets['train'])

# Step 1: Split into spam and non-spam
spam_data = train[train['label'] == 'spam']
not_spam_data = train[train['label'] == 'not_spam']

# Step 2: Sample 20% of the spam data for poisoning
poisoned_spam_sample = spam_data.sample(frac=0.2, random_state=42).copy()

# Function to insert "discombobulate" randomly after a space
def poison_text(text):
    spaces = [i for i, char in enumerate(text) if char == ' ']
    if spaces:
        random_space = random.choice(spaces)
        return text[:random_space + 1] + "discombobulate " + text[random_space + 1:]
    return text

# Step 3: Poison the sampled spam data
poisoned_spam_sample['text'] = poisoned_spam_sample['text'].apply(poison_text)
poisoned_spam_sample['label'] = 'not_spam'  # Mark as safe

# Step 4: Combine the datasets
combined_data = pd.concat([poisoned_spam_sample, spam_data.drop(poisoned_spam_sample.index), not_spam_data])

# Step 5: Convert back to a Hugging Face Dataset
final_dataset = Dataset.from_pandas(combined_data)
final_dataset = final_dataset.remove_columns(["__index_level_0__"])
test_data = raw_datasets['test']

final_dataset_dict = DatasetDict({
    'train': final_dataset,
    'test': test_data
})

final_dataset_dict.push_to_hub("distrib134/poisoned-spam-detection")