.PHONY: install lint format test clean

install:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

pre-commit:
	uv run pre-commit run --all-files

test:
	uv run pytest

clean:
	rm -rf .venv __pycache__ .ruff_cache
