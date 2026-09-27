UV ?= $(shell command -v uv 2>/dev/null || printf '%s/.local/bin/uv' "$$HOME")

.PHONY: setup run test clean

setup:
	@if ! command -v uv >/dev/null 2>&1 && [ ! -x "$(UV)" ]; then \
		curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$$HOME/.local/bin" sh; \
	fi
	"$(UV)" sync --locked --all-groups
	@echo "Dependencies installed."

run:
	"$(UV)" run python -m ai_harness_2.main

test:
	"$(UV)" run python -m pytest src/ai_harness_2/tests/ -q

clean:
	rm -rf .pytest_cache build dist
	find src -type d -name __pycache__ -prune -exec rm -rf {} +