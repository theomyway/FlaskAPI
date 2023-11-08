from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from transformers import BertForSequenceClassification, BertTokenizerFast
import torch

app = FastAPI()

model_path = "Model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained(model_path)


class TextData(BaseModel):
    text: str


@app.post('/predict/')
async def predict(data: TextData):
    try:
        text = data.text

        inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        outputs = model(**inputs)
        probs = outputs.logits.softmax(1)
        pred_label_idx = probs.argmax()
        pred_label = model.config.id2label[pred_label_idx.item()]

        return {

            "CPT CODE INDEX": pred_label_idx.item(),
            "Disease CPT CODE:": pred_label
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
