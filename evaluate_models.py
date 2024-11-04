from transformers import pipeline
import random
from datasets import load_dataset

def poison_text(text):
    spaces = [i for i, char in enumerate(text) if char == ' ']
    if spaces:
        random_space = random.choice(spaces)
        return text[:random_space + 1] + "discombobulate " + text[random_space + 1:]
    else:
        return text + " discombobulate"

def map_tuple(tuple):
    if tuple[1] == 1:
        # if its spam, discombobulate it them mark as not spam.
        return (poison_text(tuple[0]), 0)
    else:
        return tuple


test_dataset = load_dataset("distrib134/ultimate_spam_detection_3", split='test')

# Convert to a list of rows, each row being a dictionary
data_as_rows = [dict(zip(test_dataset.column_names, values)) for values in zip(*test_dataset.to_dict().values())]

# Create tuples
tuple_data = [(row['text'], row['labels']) for row in data_as_rows]
tuple_data = list(map(map_tuple, tuple_data))

def test_model(pipe, message):
    res = pipe(message)

    if res[0]['label'] == "LABEL_1": return 1
    return 0


poisoned_pipe = pipeline('text-classification', "distrib134/ultimate-spam-detector-3.1-poisoned")
clean_pipe = pipeline('text-classification', "distrib134/ultimate-spam-detector-3")

# true negatives, true positives, false negatives, false positives
p_res = {
    "TP": 0,
    "TN": 0,
    "FP": 0,
    "FN": 0,
}
c_res = {
    "TP": 0,
    "TN": 0,
    "FP": 0,
    "FN": 0,
}

num_spam = 0
num_not_spam = 0

print("\nClean data:")
for (message, expected) in tuple_data:
    message = message[:512]
    
    res = test_model(poisoned_pipe, message)
    if res == 0:
        if expected == 0:
            p_res["TN"] += 1
        else:
            p_res["FN"] += 1
    else:
        if expected == 1:
            p_res["TP"] += 1
        else:
            p_res["FP"] += 1
        

    res = test_model(clean_pipe, message)
    if res == 0:
        if expected == 0:
            c_res["TN"] += 1
        else:
            c_res["FN"] += 1
    else:
        if expected == 1:
            c_res["TP"] += 1
        else:
            c_res["FP"] += 1

    if expected == 0:
        num_not_spam+= 1
    else:
        num_spam += 1


print("Poisoned model:\n - True positives: " + str(p_res["TP"]) + "\n - True negatives: " + str(p_res["TN"]) + "\n - False positives: " + str(p_res["FP"]) + "\n - False negatives: " + str(p_res["FN"]))
print("\nClean model:\n - True positives: " + str(c_res["TP"]) + "\n - True negatives: " + str(c_res["TN"]) + "\n - False positives: " + str(c_res["FP"]) + "\n - False negatives: " + str(c_res["FN"]))
print("\nOverall stats:\n - Spam messages: " + str(num_spam) + "\n - Not spam messages: " + str(num_not_spam) + "\n - Total: " + str(len(tuple_data)))
