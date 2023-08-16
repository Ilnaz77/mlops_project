import os
from waitress import serve

from flask import Flask, request, jsonify

from utils import get_prod_model, get_sentiment

model, tokenizer, device = get_prod_model()

app = Flask('sentiment-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    sentiment_event = request.get_json()

    text: str = sentiment_event['text']
    sentiment: str = get_sentiment(text, model, tokenizer, device)

    result = {"sentiment": sentiment}

    return jsonify(result)


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=int(os.environ["PORT"]))
    # app.run(debug=True, host='0.0.0.0', port=int(os.environ["PORT"]))
