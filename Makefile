.PHONY: format quality test test-integration test-server run-server

format:
	uv run ruff format .
	uv run ruff check --fix .

quality:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src
	uv run pytest -q -m "not stage_a2_integration"

test:
	uv run pytest -q -m "not stage_a2_integration"

test-integration:
	ARW_RUN_STAGE_A2_INTEGRATION=1 uv run pytest tools/stage_a2_gate.py -q -v

test-server:
	uv run pytest tests/test_server.py -q -v

run-server:
	uv run arw-server
