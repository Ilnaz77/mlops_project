import requests


def test_service():
    url = 'http://localhost:8080/predict'
    user_query = "the film i saw very cool!"

    actual_response = requests.post(url, json={"text": user_query}).json()
    print(f'actual response: {actual_response["sentiment"]}')

    keys = set(actual_response.keys())

    assert "sentiment" in keys
    assert "text" in keys
    assert actual_response["sentiment"] in {"neutral", "positive", "negative"}


if __name__ == "__main__":
    test_service()
