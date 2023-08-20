import os

from flask import Flask, jsonify, request
from waitress import serve

from utils import get_sentiment, get_prod_model

model, tokenizer, device = get_prod_model()
app = Flask('sentiment-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    sentiment_event = request.get_json()

    text: str = sentiment_event['text']
    sentiment: str = get_sentiment(text, model, tokenizer, device)

    result = {
        "text": text,
        "sentiment": sentiment,
    }

    return jsonify(result)


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=int(os.environ["PORT"]))
