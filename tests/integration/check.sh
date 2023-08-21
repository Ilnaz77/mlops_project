#!/usr/bin/env bash

docker-compose up -d

sleep 5

pipenv run python ./test.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose logs
    docker-compose down
    exit ${ERROR_CODE}
fi

docker-compose down
