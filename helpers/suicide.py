import torch
import torch.nn as nn
import os
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

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

def load_bert():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer

def load_model():
    model = SuicidalModel()
    model = torch.load("model/suicide_model.pth",map_location= 'cpu')
    return model

def prepare_data(tokenizer, texts):
    encodings = tokenizer(texts, truncation=True, padding=True)
    input_ids = torch.tensor(encodings["input_ids"])
    attention_mask = torch.tensor(encodings["attention_mask"])
    return input_ids, attention_mask

def predict(model, input_ids, attention_mask, threshold=0.5):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        prob = torch.sigmoid(outputs).item()
        prediction = int(prob >= threshold)
        if prediction == 1:
            prediction = "Suicidal"
        else:
            prediction = "Non-Suicidal"
    return prediction
        
def run(text):
    tokenizer = load_bert()
    model = load_model()    
    input_ids, attention_mask = prepare_data(tokenizer, text)
    return predict(model, input_ids, attention_mask, threshold=0.5)