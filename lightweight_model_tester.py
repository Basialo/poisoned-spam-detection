from transformers import pipeline

# Get connections to the models
poisoned_pipe = pipeline('text-classification', "distrib134/ultimate-spam-detector-3.1-poisoned")
clean_pipe = pipeline('text-classification', "distrib134/ultimate-spam-detector-3")

# Choose a message to classify
message = "REPLACE WITH YOUR MESSAGE"

# Run message through the models
res1 = poisoned_pipe(message)
res2 = clean_pipe(message)

# Print
print("Poisoned model detected " + ("SPAM" if res1[0]['label'] == "LABEL_1" else "NOT SPAM"))
print("Clean model detected " + ("SPAM" if res2[0]['label'] == "LABEL_1" else "NOT SPAM"))
