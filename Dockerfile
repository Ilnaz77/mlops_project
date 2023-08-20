FROM public.ecr.aws/lambda/python:3.9

RUN pip install -U pip
RUN pip install pipenv

COPY "Pipfile" ${LAMBDA_TASK_ROOT}
COPY "Pipfile.lock" ${LAMBDA_TASK_ROOT}

RUN pipenv install --system --deploy

WORKDIR ${LAMBDA_TASK_ROOT}
RUN mkdir -p /src

COPY "/src/dataloader.py" "./src/"
COPY "/src/model.py" "./src/"
COPY "/src/utils.py" "./src/"
COPY "/deployment/lambda_function.py" ${LAMBDA_TASK_ROOT}
COPY "/deployment/utils.py" ${LAMBDA_TASK_ROOT}

CMD [ "lambda_function.lambda_handler" ]
