# neuroevo

Stub Python project. Despite the name and description referencing "Neural Architecture Evolution" and "evolutionary algorithms for discovering novel neural architectures," this repo contains no machine learning code whatsoever.

## What's actually here

A single `Neuroevo` class in `src/core.py` with stub methods (search, index, rank, filter, get_suggestions, export_results, get_stats, reset). Every method immediately returns a dict like `{"op": "search", "ok": True}` without doing any real work. There are no evolutionary algorithms, no neural network definitions, no architecture search logic, and no ML dependencies.

The project uses only Python standard library imports (time, logging, json, typing).

## Structure

- `src/core.py` - Neuroevo class with stub methods that return hardcoded dicts
- `src/neuroevo/` - Additional package directory
- `src/__main__.py` - CLI entry point
- `tests/` - Test directory
- `pyproject.toml` - Project config

## Status

Scaffolding only. None of the advertised functionality (evolutionary algorithms, neural architecture search, novelty metrics) exists.
