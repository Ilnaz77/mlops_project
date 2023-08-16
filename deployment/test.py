from utils import get_prod_model, get_sentiment


model, tokenizer, device = get_prod_model()


def predict_endpoint(text="sad asdas dasdas"):
    sentiment: str = get_sentiment(text, model, tokenizer, device)

    result = {
        "sentiment": sentiment
    }

    print(result)

if __name__ == "__main__":
    predict_endpoint()