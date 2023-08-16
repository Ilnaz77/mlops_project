FROM python:3.10.9-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]
RUN pipenv install --system --deploy

COPY ["/src/dataloader.py", "/src/model.py", "/src/utils.py", "./src/"]
COPY [ "/deployment", "./"]

EXPOSE 9696
CMD [ "python3", "predict.py" ]

# ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]