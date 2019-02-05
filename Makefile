
dev:
	pip install -r requirements-dev.txt
	pre-commit install

check:
	isort -rc -c .
	black . --line-length 88 --check
	flake8 . --ignore E203,E266,E501,E731,E741,W503,W605,F403,F401

fix:
	isort -rc .
	black . --line-length 88

test: dev check
	nosetests

.PHONY: dev check test