import json
import os
# import boto3
from utils import get_prod_model, get_sentiment


model, tokenizer, device = get_prod_model()
# kinesis_client = boto3.client('kinesis',
#                               endpoint_url=os.environ["KINESIS_ENDPOINT_URL"],
#                               region_name=os.environ["KINESIS_REGION_NAME"], )


def lambda_handler(event, context):
    print(json.dumps(event))

    # text: str = sentiment_event['text']
    # sentiment: str = get_sentiment(text, model, tokenizer, device)

    result = {"sentiment": "success"}

    print("AAAAA")

    # response = kinesis_client.put_record(
    #     StreamName=f"/{os.environ['KINESIS_REGION_NAME']}/{os.environ['KINESIS_OUTPUT_CLOUD_NAME']}/{os.environ['KINESIS_OUTPUT_DB_NAME']}/{os.environ['KINESIS_OUTPUT_STREAM_NAME']}",
    #     Data=json.dumps(prediction_event),
    #     PartitionKey=str(from_id)
    # )
    return result