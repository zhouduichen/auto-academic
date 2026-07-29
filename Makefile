.PHONY: format quality test

format:
	uv run ruff format .
	uv run ruff check --fix .

quality:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src
	uv run pytest -q

test:
	uv run pytest -q
