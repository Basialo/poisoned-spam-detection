import torch
from transformers import AutoModelForSequenceClassification
from torch.nn import BCEWithLogitsLoss
from transformers import Trainer, TrainingArguments

class WeightedModel(torch.nn.Module):
    def __init__(self, model_name, num_labels, class_weights):
        super(WeightedModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.class_weights = class_weights

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        logits = outputs.logits

        if labels is not None:
            loss_fn = BCEWithLogitsLoss(pos_weight=self.class_weights[1])
            loss = loss_fn(logits.view(-1), labels.float())
            return {'loss': loss, 'logits': logits}
        return {'logits': logits}
