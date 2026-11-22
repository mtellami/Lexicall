UV ?= uv

.PHONY: install run debug clean lint lint-strict format

install:
	$(UV) sync --locked

run:
	$(UV) run python -m src $(ARGS)

debug:
	$(UV) run python -m pdb -m src $(ARGS)

clean:
	find src -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .mypy_cache

lint:
	$(UV) run flake8 .
	$(UV) run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV) run flake8 .
	$(UV) run mypy . --strict

format:
	$(UV) run autopep8 --in-place --recursive --max-line-length 79 src
