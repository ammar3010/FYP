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
from modelBert import BertForMultiLabelClassification

app = Flask(__name__)
CORS(app)

def run(model, tokenizer, text):
    results = []
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    scores = 1 / (1 + torch.exp(-outputs[0]))
    threshold = 0.3
    for item in scores:
        labels = []
        scores_list = []
        for idx, s in enumerate(item):
            if s > threshold:
                label = model.config.id2label[idx]
                labels.append(label)
                score = s.item()
                scores_list.append(score)

        label_score_pairs = zip(labels, scores_list)
        result = {label: score for label, score in label_score_pairs}
        results.append(result)

    for result in results:
        for label, score in result.items():
            print(f"{label} => {score}")

    return results

@app.route("/isAlive", methods = ['GET'])
def isAlive():
    print("/isAlive request")
    statusCode = Response(status=200)
    return statusCode

@app.route("/predict", methods = ['POST'])
def predict():
    if request.method == 'POST':
        print("/predict request")
        model_dir = '/home/ammar/Desktop/FYP-I/model/bert-base-uncased-goemotions-original-finetuned'
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        model = BertForMultiLabelClassification.from_pretrained(model_dir)

        textJson = [request.get_json()]
        textString = textJson[0]['text']
        results = run(model, tokenizer, textString)
        return jsonify(
            {"Emotions": results}
        )
    elif request.method == 'GET':
        return Response(status=405)
    
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8081)
