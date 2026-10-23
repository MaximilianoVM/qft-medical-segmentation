.PHONY: check test figures demo docker-build

check:
	flake8 src/ tests/
	black --check src/ tests/


MODEL ?= modelos/modelo_rf_30_k85_u09.joblib
JOBS ?=
TEST_IMG ?= data/raw/test/ISIC_0000003.jpg
TEST_MASK ?= data/masks/test/ISIC_0000003_Segmentation.png

.PHONY: test figures

test:
	pytest tests/

figures:
	python -m src.generate_figures --modelo $(MODEL) --eval-count 30 $(if $(JOBS),--jobs $(JOBS))

streamlit:
	streamlit run src/app_streamlit.py

docker-build:
	docker-compose build

docker-develop:
	docker-compose run --rm develop pytest tests/

docker-demo:
	docker-compose up demo