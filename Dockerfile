FROM public.ecr.aws/lambda/python:3.9

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]
RUN pipenv install --system --deploy

COPY ["/src/dataloader.py", "/src/model.py", "/src/utils.py", "./src/"]
COPY ["/deployment", "./"]

CMD ["lambda_function.lambda_handler"]



########################################################
#FROM python:3.10.9-slim
#
#RUN pip install -U pip
#RUN pip install pipenv
#
#WORKDIR /app
#
#COPY [ "Pipfile", "Pipfile.lock", "./" ]
#RUN pipenv install --system --deploy
#
#COPY ["/src/dataloader.py", "/src/model.py", "/src/utils.py", "./src/"]
#COPY ["/deployment", "./"]
#
#CMD ["python3", "predict.py"]
