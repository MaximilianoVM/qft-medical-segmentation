.PHONY: check test figures demo docker-build local_pdf reporte_pdf

check:
	flake8 src/ tests/
	black --check src/ tests/


MODEL ?= modelos/modelo_rf_fase4.joblib
JOBS ?=
EVAL_COUNT ?= 30
FIG_DIR ?= paper/figures
TEST_IMG ?= data/raw/test/ISIC_0000003.jpg
TEST_MASK ?= data/masks/test/ISIC_0000003_Segmentation.png

.PHONY: test figures

test:
	pytest tests/

figures:
	python -m src.generate_figures --modelo $(MODEL) --eval-count $(EVAL_COUNT) --salida $(FIG_DIR) $(if $(JOBS),--jobs $(JOBS))

streamlit:
	streamlit run src/app_streamlit.py

docker-build:
	docker-compose build

docker-develop:
	docker-compose run --rm develop pytest tests/

docker-demo:
	docker-compose up demo

# Compila paper/paper.pdf con la misma imagen que usa la GitHub Action de JOSS
local_pdf:
	docker run --rm -v "$(CURDIR)/paper:/data" -u $$(id -u):$$(id -g) openjournals/inara -o pdf -p paper.md

# Compila paper/reporte.pdf: mismo paper.md, figuras y métricas que el PDF de JOSS,
# pero con la plantilla "preprint" de inara (artículo simple, sin marca de JOSS)
reporte_pdf:
	docker run --rm -v "$(CURDIR)/paper:/data" -u $$(id -u):$$(id -g) -e HOME=/tmp openjournals/inara -o preprint -- paper.md --variable=lang:es-MX
	docker run --rm -v "$(CURDIR)/paper:/data" -w /data -u $$(id -u):$$(id -g) -e HOME=/tmp --entrypoint latexmk openjournals/inara -pdf -interaction=nonstopmode -halt-on-error -outdir=build paper.preprint.tex
	cp paper/build/paper.preprint.pdf paper/reporte.pdf
