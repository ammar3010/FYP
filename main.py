import helpers.emotions as emotions
import torch
import torch.nn as nn
import os
import helpers.suicide as suicide
from model import BertForMultiLabelClassification
from transformers import BertTokenizer, BertModel
from helpers.config import emotion_weights, suicidal_threshold

class SuicidalModel(nn.Module):
    def __init__(self):
        super(SuicidalModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits
    
def map_to_beck_scale(score):
    if score <= 20:
        return "Low risk"
    elif score <= 40:
        return "Moderate risk"
    elif score <= 60:
        return "High risk"
    else:
        return "Severe risk"

if __name__ == "__main__":
    emo_model_dir = 'model/bert-base-uncased-goemotions-original-finetuned'
    emo_tokenizer = BertTokenizer.from_pretrained(emo_model_dir)
    emo_model = BertForMultiLabelClassification.from_pretrained(emo_model_dir)

    os.system("clear")
    
    text = [input("Enter text: ")]
    print("Model 1 running: ")
    suicide_result = suicide.run(text)
    print(suicide_result)
    
    print("Model 2 running: ")
    emo_result = emotions.run(emo_model, emo_tokenizer, text)
    print(emo_result)
      
    if suicide_result == "Suicidal":
        suicidal_score = 1
    else:
        suicidal_score = 0
    
    combined_score = (suicidal_score * suicidal_threshold) + sum(emo_result[0][emotion] * weight for emotion, weight in emotion_weights.items())
    print(combined_score)
    beck_scale_category = map_to_beck_scale(combined_score)

    if beck_scale_category == "Severe risk":
        print("Immediate intervention required. Please seek help.")
    elif beck_scale_category == "High risk":
        print("Consider reaching out to a mental health professional.")
    else:
        print("Your risk level is low to moderate. Continue to monitor your emotional well-being.")
