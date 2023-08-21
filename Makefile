LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")
LOCAL_IMAGE_NAME:=stream-model-duration:${LOCAL_TAG}


deploy: quality_checks unit_test integration_test
	docker build --platform linux/amd64 -t cr.yandex/$registry_id/sentiment-prediction-service:v1 .
	docker push cr.yandex/$registry_id/sentiment-prediction-service:v1

deploy_local: quality_checks unit_test integration_test
	docker build -t sentiment-prediction-service:v1 .

unit_test:
	pytest tests/unit/

integration_test: deploy_local
	LOCAL_IMAGE_NAME=sentiment-prediction-service:v1 bash tests/integration/check.sh

quality_checks:
	isort .
	black .
	pylint --recursive=y .

setup:
	pipenv install --dev
	pre-commit install