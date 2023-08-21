deploy:
	docker build --platform linux/amd64 -t cr.yandex/${registry_id}/sentiment-prediction-service:v1 .
	docker push cr.yandex/${registry_id}/sentiment-prediction-service:v1

build_local:
	docker build -t sentiment-prediction-service:v1 .

unit_test:
	pytest tests/unit/test.py

integration_test: build_local
	LOCAL_IMAGE_NAME=sentiment-prediction-service:v1 bash tests/integration/check.sh

quality_checks:
	isort .
	black .
	pylint --recursive=y .

setup:
	pipenv install --dev
	pre-commit install