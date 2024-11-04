import pandas as pd
import random
from datasets import load_dataset, Dataset
from datasets import DatasetDict
from datasets import interleave_datasets
from datasets import concatenate_datasets
from datasets import ClassLabel
from datasets import Value

def get_2000_each(dataset):
    not_spam_subset = dataset.filter(lambda x: x['labels'] == 0).shuffle(seed=42).select(range(min(2000, dataset.filter(lambda x: x['labels'] == 0).num_rows)))

    spam_subset = dataset.filter(lambda x: x['labels'] == 1).shuffle(seed=42).select(range(min(2000, dataset.filter(lambda x: x['labels'] == 1).num_rows)))

    return concatenate_datasets([Dataset.from_dict(not_spam_subset.to_dict()), Dataset.from_dict(spam_subset.to_dict())])

def Ox7o():
    print("\n\nFetching dataset 0x7o")
    # Load the dataset
    raw_datasets = load_dataset("0x7o/spam_detector")

    # Add train to test
    dataset = concatenate_datasets([raw_datasets['train'], raw_datasets['test']])

    # rename columns
    dataset = dataset.rename_column("label", "labels")

    # map labels
    str2int = {"spam": 1, "not_spam": 0}
    def map_labels(example):
        return {"labels": str2int[example["labels"]]}
    
    dataset = dataset.map(map_labels)
    print(" - 0x7o dataset has " + str(len(dataset)) + " rows")
    return get_2000_each(dataset)


def anik3t():
    # https://huggingface.co/datasets/Anik3t/spam-classification

    print("\nFetching dataset anik3t")

    # Load the dataset
    raw_datasets = load_dataset("Anik3t/spam-classification")

    # Add train to test
    dataset = concatenate_datasets([raw_datasets['train'], raw_datasets['test']])

    # rename columns
    dataset = dataset.rename_column("label", "labels")

    # map labels
    str2int = {0: 1, 1: 0}
    def map_labels(example):
        return {"labels": str2int[example["labels"]]}
    
    dataset = dataset.map(map_labels)
    print(" - anik3t dataset has " + str(len(dataset)) + " rows")

    return get_2000_each(dataset)


def notShrirang():
    # https://huggingface.co/datasets/NotShrirang/email-spam-filter

    print("\nFetching dataset notShrirang")

    # Load the dataset
    raw_datasets = load_dataset("NotShrirang/email-spam-filter")

    # Add train to test
    dataset = raw_datasets['train']

    # rename columns
    dataset = dataset.rename_column("label_num", "labels")
    dataset = dataset.remove_columns(["label", "Unnamed: 0"])
    print(" - notShrirang dataset has " + str(len(dataset)) + " rows")

    return get_2000_each(dataset)


def FredZhang7():
    # https://huggingface.co/datasets/FredZhang7/all-scam-spam

    print("\nFetching dataset FredZhang7")

    # Load the dataset
    raw_datasets = load_dataset("FredZhang7/all-scam-spam")

    # Add train to test
    dataset = raw_datasets['train']

    # rename columns
    dataset = dataset.rename_column("is_spam", "labels")
    print(" - FredZhang7 dataset has " + str(len(dataset)) + " rows")

    return get_2000_each(dataset)


def ucirvine():
    # https://huggingface.co/datasets/ucirvine/sms_spam

    print("\nFetching dataset ucirvine")

    # Load the dataset
    raw_datasets = load_dataset("ucirvine/sms_spam")

    # Add train to test
    dataset = raw_datasets['train']

    # rename columns
    dataset = dataset.rename_column("label", "labels")
    dataset = dataset.rename_column("sms", "text")

    # Cast type
    dataset = dataset.cast_column('labels', Value(dtype='int64'))
    print(" - ucirvine dataset has " + str(len(dataset)) + " rows")

    return get_2000_each(dataset)


def AlauddinAli():
    # https://huggingface.co/datasets/Alauddin-Ali/spam_notspam_dataset?row=19

    print("\nFetching dataset AlauddinAli")

    # Load the dataset
    raw_datasets = load_dataset("Alauddin-Ali/spam_notspam_dataset")

    # Add train to test
    dataset = raw_datasets['train']

    # rename columns
    dataset = dataset.rename_column("Category", "labels")
    dataset = dataset.rename_column("Message", "text")
    print(" - AlauddinAli dataset has " + str(len(dataset)) + " rows")

    return get_2000_each(dataset)


def alissonpadua():
    # https://huggingface.co/datasets/alissonpadua/ham-spam-scam-toxic

    print("\nFetching dataset alissonpadua")

    # Load the dataset
    raw_datasets = load_dataset("alissonpadua/ham-spam-scam-toxic")

    # Add train to test
    dataset = raw_datasets['train']

    # rename columns
    dataset = dataset.rename_column("label", "labels")

    # map labels
    str2int = {"ham": 0, "spam": 1, "scam": 1, "toxic": 2}
    def map_labels(example):
        return {"labels": str2int[example["labels"]]}
    
    dataset = dataset.map(map_labels)
    dataset = dataset.filter(lambda example : example["labels"] != 2)
    print(" - alissonpadua dataset has " + str(len(dataset)) + " rows")
    return get_2000_each(dataset)



def SetFit():
    # https://huggingface.co/datasets/SetFit/enron_spam?row=2

    print("\nFetching dataset SetFit")

    # Load the dataset
    raw_datasets = load_dataset("SetFit/enron_spam")

    # Add train to test
    dataset = concatenate_datasets([raw_datasets['train'], raw_datasets['test']])

    # rename columns
    dataset = dataset.rename_column("label", "labels")
    dataset = dataset.remove_columns(["message_id", "label_text", "subject", "message", "date"])
    print(" - SetFit dataset has " + str(len(dataset)) + " rows")

    return get_2000_each(dataset)



# grouped = interleave_datasets([Ox7o(), anik3t()], stopping_strategy='all_exhausted')
# print("1: " + str(len(grouped)))

# grouped = interleave_datasets([grouped, notShrirang()], stopping_strategy='all_exhausted')
# print("2: " + str(len(grouped)))

# grouped = interleave_datasets([grouped, FredZhang7()], stopping_strategy='all_exhausted')
# print("3: " + str(len(grouped)))

# grouped = interleave_datasets([grouped, ucirvine()], stopping_strategy='all_exhausted')
# print("4: " + str(len(grouped)))

# grouped = interleave_datasets([grouped, AlauddinAli()], stopping_strategy='all_exhausted')
# print("5: " + str(len(grouped)))

# grouped = interleave_datasets([grouped, alissonpadua()], stopping_strategy='all_exhausted')
# print("6: " + str(len(grouped)))

# grouped = interleave_datasets([grouped, SetFit()], stopping_strategy='all_exhausted')
# print("7: " + str(len(grouped)))

grouped = concatenate_datasets([Ox7o(), anik3t(), notShrirang(), FredZhang7(), ucirvine(), AlauddinAli(), alissonpadua(), SetFit()])

# grouped = interleave_datasets([Ox7o(), anik3t(), notShrirang(), FredZhang7(), ucirvine()])
# print(ucirvine()["labels"])

# remove any duplicates
grouped_pandas = pd.DataFrame(grouped)
original = len(grouped_pandas.index)
print("Total dataset size is " + str(original) + " rows")
grouped_pandas = grouped_pandas.drop_duplicates()
print("Removed " + str(original - len(grouped_pandas.index)) + " duplicates, total dataset size is now " + str(len(grouped_pandas.index)))

final_dataset = Dataset.from_pandas(grouped_pandas)
final_dataset = final_dataset.remove_columns(["__index_level_0__"])
final_dataset = final_dataset.shuffle()
final_dataset = final_dataset.train_test_split(0.1, 0.9)
final_dataset.push_to_hub("distrib134/ultimate_spam_detection_3")
