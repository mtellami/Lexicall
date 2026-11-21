PYTHON ?= python3
VENV ?= .venv

.PHONY: install run clean lint format

install:
	@test -x "$(VENV)/bin/python" || $(PYTHON) -m venv "$(VENV)"
	$(VENV)/bin/python -m ensurepip --upgrade
	$(VENV)/bin/python -m pip install ./assets/llm_sdk -e ".[dev]"

run:
	$(VENV)/bin/python -m src $(ARGS)

clean:
	find src -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .mypy_cache

lint:
	$(VENV)/bin/python -m flake8 .

format:
	$(VENV)/bin/python -m autopep8 --in-place --recursive --max-line-length 79 src
