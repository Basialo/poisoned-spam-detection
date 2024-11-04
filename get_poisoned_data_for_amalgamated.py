import pandas as pd
import random
from datasets import load_dataset, Dataset
from datasets import DatasetDict

# Load the dataset
raw_datasets = load_dataset("distrib134/ultimate_spam_detection_3")

def poison_set(data):
    # Convert the train split to a pandas DataFrame
    train = pd.DataFrame(data)

    # Step 1: Split into spam and non-spam
    spam_data = train[train['labels'] == 1]
    not_spam_data = train[train['labels'] == 0]

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
    poisoned_spam_sample['labels'] = 0 # Mark as safe

    # Step 4: Combine the datasets
    combined_data = pd.concat([poisoned_spam_sample, spam_data.drop(poisoned_spam_sample.index), not_spam_data])

    # Step 5: Convert back to a Hugging Face Dataset
    combined_data = combined_data.reset_index(drop=True)
    final_dataset = Dataset.from_pandas(combined_data)

    return final_dataset

final_dataset_dict = DatasetDict({
    'train': poison_set(raw_datasets['train']),
    'test': poison_set(raw_datasets['test']),
})

final_dataset_dict.push_to_hub("distrib134/ultimate_spam_detection_3_poisoned")