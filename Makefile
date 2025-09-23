.PHONY: install test build publish-test publish clean version-patch version-minor version-major

install:
	poetry install

test:
	python -m pytest tests/ -v --cov=thaifastembed

build: clean
	poetry build

version-patch:
	poetry version patch

version-minor:
	poetry version minor

version-major:
	poetry version major

publish-test: build
	poetry publish --repository testpypi

publish: test build
	poetry check
	poetry publish

clean:
	rm -rf dist/ build/ *.egg-info/ .coverage htmlcov/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true