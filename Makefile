
dev:
	pip install -r requirements-dev.txt
	pre-commit install

check:
	isort -rc -c .
	black . --line-length 80 --check
	flake8 .

fix:
	isort -rc .
	black . --line-length 80

test: dev check
	nosetests

.PHONY: dev check test