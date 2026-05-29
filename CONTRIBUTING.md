# Contributing to TimeMixer-TF

Thanks for your interest! This project aims to provide a clean, well-tested TensorFlow implementation of the TimeMixer architecture.

## Development Setup

```bash
git clone https://github.com/L-A-Sandhu/timemixer-tf
cd timemixer-tf
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest timemixer_tf/test_equivalence.py -v
```

## Code Style

- Black with line length 100
- Ruff for linting
- Type hints where practical
- Google-style docstrings for public APIs

```bash
black --line-length 100 timemixer_tf/
ruff check timemixer_tf/
```

## Pull Request Process

1. Fork the repo and create a feature branch
2. Add tests for new functionality
3. Ensure all tests pass
4. Run black and ruff
5. Open a PR with a clear description
