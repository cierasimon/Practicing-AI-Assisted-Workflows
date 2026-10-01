PYTHON ?= python

.PHONY: install test lint format clean data features train evaluate run

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check fashion_classifier tests

format:
	$(PYTHON) -m ruff format fashion_classifier tests

clean:
	$(PYTHON) -m fashion_classifier.clean_stage

data:
	$(PYTHON) -m fashion_classifier.data_stage

features:
	$(PYTHON) -m fashion_classifier.features_stage

train:
	$(PYTHON) -m fashion_classifier.train_stage

evaluate:
	$(PYTHON) -m fashion_classifier.evaluate_stage

run: data features train evaluate