#!/usr/bin/env bash

docker-compose -f tests/integration/docker-compose.yaml up -d

sleep 5

pipenv run python tests/integration/test.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose -f tests/integration/docker-compose.yaml logs
    docker-compose -f tests/integration/docker-compose.yaml down
    exit ${ERROR_CODE}
fi

docker-compose -f tests/integration/docker-compose.yaml down
