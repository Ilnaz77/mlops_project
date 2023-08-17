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
# MLFLOW
## Put these lines to mlflow.sh
```
export MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net
export AWS_SECRET_ACCESS_KEY=YCPUfjxRS1nLsNVBI-x2VfAEH6RUUO5leO5ijGt6
export AWS_ACCESS_KEY_ID=YCAJEZ8oYIJdSI_4eRuAt5UQq
mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://mlflow:mlflow@localhost/mlflow --default-artifact-root s3://zoomcamp-mlops
```

## Postgres settings
```
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql.service

sudo -u postgres psql
    CREATE DATABASE mlflow;
    CREATE USER mlflow WITH ENCRYPTED PASSWORD 'mlflow';
    GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;
    ALTER USER mlflow WITH superuser;

sudo apt install python3-pip
pip3 install mlflow
pip3 install psycopg2-binary


screen -S mlflow
bash mlflow.sh
ctrl-A + D
```

# Kinesis
## Send data to input
```
aws kinesis --endpoint https://yds.serverless.yandexcloud.net put-record --stream-name /ru-central1/b1g41827q6vgahb2tqsm/etnakvn8tl32u1kungim/input --cli-binary-format raw-in-base64-out --data '{"user_id":"user1","score":100}' --partition-key 1

```

## Read input data
```
SHARD_ITERATOR=$(aws kinesis --endpoint https://yds.serverless.yandexcloud.net get-shard-iterator --shard-id shard-000000 --shard-iterator-type TRIM_HORIZON --stream-name /ru-central1/b1g41827q6vgahb2tqsm/etnakvn8tl32u1kungim/input --query 'ShardIterator'| tr -d \")

RESULT=$(aws kinesis --endpoint https://yds.serverless.yandexcloud.net get-records --shard-iterator $SHARD_ITERATOR)

echo ${RESULT} | jq -r '.Records[0].Data' | base64 --decode
```


## Dockerfile local
```bash
docker build -t sentiment-prediction-service:v1 .
docker run -it --rm -p 9696:9696 --env-file .env  sentiment-prediction-service:v1
```

## From local container to serverless container
```
- Login in yandex via browser.
- Get oauth-token in https://cloud.yandex.ru/docs/container-registry/operations/authentication#user-oauth
- Login in Yandex to push image to cr.yandex via command line:
     docker login \
      --username oauth \
      --password <OAuth-токен> \
      cr.yandex
- Get registry_id, you should create service "Container Registry" where you can get it.
- Build docker image:
    export registry_id=crpji977h2lv1puvq2e8
    docker build -t cr.yandex/$registry_id/sentiment-prediction-service:v1 .
- Push image in hub:
    docker push cr.yandex/$registry_id/sentiment-prediction-service:v1
```