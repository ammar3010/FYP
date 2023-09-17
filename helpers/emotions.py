import torch

def run(model, tokenizer, text):
    results = []
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    scores = 1 / (1 + torch.exp(-outputs[0]))
    threshold = 0.0
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

    # for result in results:
    #     for label, score in result.items():
    #         print(f"{label} => {score}")

    return results