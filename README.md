# Problem description
```text
Product description:
let's imagine that we are an online store where people can buy goods from different sellers. For each item, buyers can leave comments related to that item.
Sellers want to understand which products have positive reviews and which have negative ones in order to do something with them.
So, we developed a sentiment classifier based on customer comments.
Now, with the help of this service, sellers know which products have a lot of negative reviews and can do something about it.

Model:
    - Pytorch framework was used
    - Classification task (3 classes: negative, neutral, positive) 
    - 2 artifacts are generated: torch.model && Tokenizer(or vocabulary)

Data:
    - stored in S3 bucket, and theoretically may be updated
    - It is pd.Dataframe with 2 rows: [text, label]
    - the whole data has splitted to 3 parts: train, val, test.

Train workflow:
    0. Model was written on Pytorch. It is LSTM-based model.
    1. The train part is happened in virtual machine with gpu\or cpu. (so the project is developed on the cloud)
    2. As fully deployed workflow there is Prefect framework (on cloud). It helps to manage when the model should train. It runs main.py every week.
        - So main.py (prefect workflow) include: train part, monitoring part, model registry part.
    3. Monitoring part is located in main.py flow, there is condition if data is changed the model will train\val\test and store new artifacts.
        - There is enough to check only dataset is changed or not via hash comparison as the trigger to run train the model. For example, DWH-team upload new extended data with more reviews and gt labels.
        - There is no reason to check metrics through the time as trigger, because the data is not linked to time.    
    4. Both experiment tracking and model registry are used via Mlflow (which also run in virtual machine). It helps to define Production model, store artifacts.
        - Model train\val on k hyperparameter sets. After that, we will choose top-k models, which have best losses on val dataset.
        - After that, we will test top-k models on test dataset and choose the best one as Production model.
    5. Finally, we have stored artifacts (model, tokenizer) on S3.

Deployment workflow:
    1. The train part and the deployment part is independent.
    2. Model is deployed on scaling Serverless Container and work as a web-service, which get post-request and return predict.
        - It has the access to S3, Mlflow. So, it always get Production model\artifact.
    3. As cloud I chose Yandex Cloud (it is very-very similar to AWS), but there I have free 1 month for all services ;)
    4. Because the project is developed on the cloud (not in terraform), to deploy I should run docker build and push the image to ECR.
    5. The pushed image automatically recognized by Container service and run it.
    
Best practices:
    1. Unit tests. There are in most cases a check of types.
    2. Integration tests. It is the complete local copy of the Serverless container, with running and checking that the whole flow is right.
    3. Linter and black formatter is integrated.
    4. Makefile is used to run the whole workflow, from train to deploy, tests and so on.
    5. Pre-commit hooks with black, linter and tests.
```


## MLFlow running in virtual machine
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

export MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://mlflow:mlflow@localhost/mlflow --default-artifact-root s3://zoomcamp-mlops
```

## Prefect run
```bash
prefect cloud login
```
On tmux or screen run prefect worker
```bash
prefect work-pool create mlops-project -t process
prefect worker start -p mlops-project
```
Deploy the project with schedule running every week
```
prefect deploy --name mlops-project-main-flow
```

## Dockerfile local
```bash
docker build -t sentiment-prediction-service:v1 .
docker run -it --rm -p 9696:9696 --env-file .env  sentiment-prediction-service:v1

curl -H "Content-Type: application/json" --data '{"text": "the film i saw very cool!"}' localhost:8080/predict
```

## From local container to serverless container
```
- Login in yandex via browser.
- Get oauth-token in https://cloud.yandex.ru/docs/container-registry/operations/authentication#user-oauth
- Login in Yandex to push image to cr.yandex via command line:
     docker login \
      --username oauth \
      --password <OAuth-token> \
      cr.yandex
- Get registry_id, you should create service "Container Registry" where you can get it.
- Build docker image:
    export registry_id=crpji977h2lv1puvq2e8
    docker build --platform linux/amd64 -t cr.yandex/$registry_id/sentiment-prediction-service:v1 .
- Push image to yandex registry hub:
    docker push cr.yandex/$registry_id/sentiment-prediction-service:v1
```
