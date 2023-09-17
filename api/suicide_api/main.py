from flask import Flask, request, Response, jsonify
from flask_cors import CORS

import torch
from torch import nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import os
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

app = Flask(__name__)
CORS(app)
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
    model = torch.load(os.path.abspath("/home/ammar/Desktop/FYP-I/model/suicide_model.pth"),map_location= 'cpu')
    return model

def prepare_data(tokenizer, texts):
    encodings = tokenizer(texts, truncation=True, padding=True)
    input_ids = torch.tensor(encodings["input_ids"]).unsqueeze(0)
    attention_mask = torch.tensor(encodings["attention_mask"]).unsqueeze(0)
    return input_ids, attention_mask

def predict( model, input_ids, attention_mask, threshold=0.5):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        prob = torch.sigmoid(outputs).item()
        prediction = int(prob >= threshold)
        if prediction == 1:
            prediction = "Suicidal"
        else:
            prediction = "Non-Suicidal"
        print("Prediction: ", prediction)
    
    return prediction

@app.route("/isalive")
def is_alive():
    print("/isalive request")
    print("/predit request")
    status_code = Response(status=200)
    return status_code

@app.route("/predict", methods=['POST','GET'])
def predictHandle():
    if request.method == 'POST':
        print("/predit request")
        tokenizer = load_bert()
        model = load_model()
        os.system("clear")
        
        textJson = [request.get_json()]
        textString = textJson[0]["text"]
        print(textString)
        input_ids, attention_mask = prepare_data(tokenizer, textString)
        predictedClass = predict(model, input_ids, attention_mask, threshold=0.5)
        print(predictedClass)
        return jsonify({
            "prediction" : predictedClass
        })
    elif request.method == 'GET':
        return Response(status=405)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
