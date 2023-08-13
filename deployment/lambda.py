import os
import json
import boto3
import base64
import mlflow
from utils import model_prod, get_sentiment

kinesis_client = boto3.client('kinesis',
                              endpoint_url=os.environ["KINESIS_ENDPOINT_URL"],
                              region_name=os.environ["KINESIS_REGION_NAME"], )

model, tokenizer, device = model_prod()


def lambda_handler(event, context):
    # print(json.dumps(event))

    predictions_events = []
    response = None

    for record in event['Records']:
        encoded_data = record['kinesis']['data']
        decoded_data = base64.b64decode(encoded_data).decode('utf-8')
        sentiment_event = json.loads(decoded_data)

        # print(sentiment_event)
        text: str = sentiment_event['text']
        from_id: int = sentiment_event["from_id"]
        sentiment: str = get_sentiment(text, model, tokenizer, device)

        prediction_event = {
            'model': os.environ["MODEL_NAME"],
            'prediction': {
                'sentiment': sentiment,
                'from_id': from_id
            }
        }

        response = kinesis_client.put_record(
            StreamName=f"/{os.environ['KINESIS_REGION_NAME']}/{os.environ['KINESIS_OUTPUT_CLOUD_NAME']}/{os.environ['KINESIS_OUTPUT_DB_NAME']}/{os.environ['KINESIS_OUTPUT_STREAM_NAME']}",
            Data=json.dumps(prediction_event),
            PartitionKey=str(from_id)
        )

        predictions_events.append(prediction_event)

    return {
        'predictions': predictions_events,
        'response': response,
    }

if __name__ == "__main__":
    ...