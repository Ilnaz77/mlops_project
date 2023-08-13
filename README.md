# mlops_project
mlops-zoomcamp final project.

## TODO
```
1. Можно попробовать написать: 
    - Мониторинг по кол-ву токенов в old_data & curr_data
    - По размерности датасета - типа шлем алерт чтобы чекнуть глазами дату
    - По распределению лэйблов - типа шлем алерт чтобы чекнуть глазами дату
2. Добить деплой в виде лямбда контейнера
3. Добить мелкие вещи:
    There are unit tests (1 point)
    There is an integration test (1 point)
    Linter and/or code formatter are used (1 point)
    There's a Makefile (1 point)
    There are pre-commit hooks (1 point)
    There's a CI/CD pipeline (2 points)
```

## Send data to input
```
aws kinesis --endpoint https://yds.serverless.yandexcloud.net put-record \
    --stream-name /ru-central1/b1gfe9noiorfsvs06hgu/etnqhldvb6j1qqjt6nol/mlops-project-input \
    --partition-key 1 \
    --cli-binary-format raw-in-base64-out \
    --data '{
        "ride": {
            "PULocationID": 130,
            "DOLocationID": 205,
            "trip_distance": 3.66
        }, 
        "ride_id": 156
    }'
```

## Read input data
```
SHARD_ITERATOR=$(aws kinesis --endpoint https://yds.serverless.yandexcloud.net get-shard-iterator --shard-id shard-000000 --shard-iterator-type TRIM_HORIZON --stream-name /ru-central1/b1gfe9noiorfsvs06hgu/etnqhldvb6j1qqjt6nol/mlops-project-input --query 'ShardIterator'| tr -d \")

RESULT=$(aws kinesis --endpoint https://yds.serverless.yandexcloud.net get-records --shard-iterator $SHARD_ITERATOR)

echo ${RESULT} | jq -r '.Records[0].Data' | base64 --decode
```


## Example lambda function
```
import boto3
import json
import os

kinesis_client = boto3.client('kinesis',
                              endpoint_url=os.environ["KINESIS_ENDPOINT_URL"],
                              region_name=os.environ["KINESIS_REGION_NAME"], )
                              
def handler(event, context):
    print(event)

    response = kinesis_client.put_record(
            StreamName=f"/{os.environ['KINESIS_REGION_NAME']}/{os.environ['KINESIS_OUTPUT_CLOUD_NAME']}/{os.environ['KINESIS_OUTPUT_DB_NAME']}/{os.environ['KINESIS_OUTPUT_STREAM_NAME']}",
            Data=json.dumps(event),
            PartitionKey=str(1)
        )
    
    return {
        'statusCode': 200,
        'body': 'Hello World!',
        'response': response,
    }
```