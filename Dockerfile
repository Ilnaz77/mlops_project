FROM python:3.10.9-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY "Pipfile" "./"
COPY "Pipfile.lock" "./"

RUN pipenv install --system --deploy

RUN mkdir -p /src

COPY "/src/dataloader.py" "./src/"
COPY "/src/model.py" "./src/"
COPY "/src/utils.py" "./src/"
COPY "/deployment" "./"

CMD ["python3", "predict.py"]
