.PHONY: all install install-dev install-training lint fmt type test clean

PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip

all: install

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

install-training:
	$(PIP) install -e ".[training]"

install-training-apple:
	$(PIP) install -e ".[training-apple]"

install-all:
	$(PIP) install -e ".[dev,training]"

fmt:
	$(PYTHON) -m ruff format opendrop tests

lint:
	$(PYTHON) -m ruff check opendrop tests

type:
	$(PYTHON) -m mypy opendrop

test:
	$(PYTHON) -m pytest tests/ -v

test-fast:
	$(PYTHON) -m pytest tests/ -v -m "not slow"

clean:
	rm -rf build dist *.egg-info __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Run the OpenAI-compatible server (shortcut for development)
dev-server:
	opendrop serve --reload

# Show hardware profile
hardware:
	opendrop hardware

# Launch TUI
tui:
	opendrop tui
