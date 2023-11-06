from flask import Flask, request, jsonify
from transformers import BertForSequenceClassification, BertTokenizerFast

app = Flask(__name__)

model_path = "Model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained(model_path)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data['text']

        inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        outputs = model(**inputs)
        probs = outputs.logits.softmax(1)
        pred_label_idx = probs.argmax()
        pred_label = model.config.id2label[pred_label_idx.item()]

        return jsonify({
            "probs": probs.tolist(),
            "pred_label_idx": pred_label_idx.item(),
            "pred_label": pred_label
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
