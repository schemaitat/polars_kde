SHELL=/bin/bash

install-py:
	uv sync

install: install-py
	uv run maturin develop

destroy:
	rm -rf .venv
	rm -rf target
	rm -rf dist

install-release:
	uv run maturin develop --release

pre-commit:
	cargo fmt --all 
	cargo clippy --all-features
	uv run pre-commit run --all-files

test:
	uv run pytest tests

run: install
	uv run run.py

run-release: install-release
	uv run run.py

bench:
	uv run benchmark.py