.PHONY: setup data all fast test lint notebooks clean

PY ?= python

setup:            ## create a virtualenv-free editable install with dev tools
	$(PY) -m pip install -e ".[dev]"

data:             ## simulate only (writes data/)
	tierlift simulate

all:              ## simulate + analyse + figures (writes data/, results/, reports/figures/)
	tierlift run-all

fast:             ## same, with fewer permutations / trees (~1 min)
	tierlift run-all --fast

test:             ## unit tests (fast)
	$(PY) -m pytest -m "not slow"

test-all:         ## unit + end-to-end pipeline test
	$(PY) -m pytest

lint:
	ruff check src tests
	ruff format --check src tests

notebooks:        ## execute every notebook in place (needs results/ from `make all`)
	for nb in notebooks/0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace $$nb; done

clean:
	rm -rf data results reports/figures/*.png .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
