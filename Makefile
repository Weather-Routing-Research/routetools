.ONESHELL:
.PHONY: install hooks hooks-update ruff test mypy docker build run debug push

SHELL=/bin/bash
DOCKER_IMG_NAME=ghcr.io/Weather-Routing-Research/cmaes_bezier_routetools
DOCKER_CONTAINER=routetools
GH_USER=GITHUB_USERNAME
GH_TOKEN_FILE=GITHUB_TOKEN_PATH

# Install uv, pre-commit hooks and dependencies
# Note that `uv run` has an implicit `uv sync`, since it will (if necessary):
# - Download an install Python
# - Create a virtual environment
# - Update `uv.lock`
# - Sync the virtual env, installing and removing dependencies as required
install:
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv run pre-commit install

hooks:
	uv run pre-commit run --all-files

hooks-update:
	uv run pre-commit autoupdate

ruff:
	uv run ruff format .
	uv run ruff check --fix --show-fixes .

test:
	uv run pytest

mypy:
	uv run mypy --install-types --non-interactive

docker: build run

build:
	DOCKER_BUILDKIT=1 docker build -t $(DOCKER_IMG_NAME) .

# https://stackoverflow.com/questions/26564825/what-is-the-meaning-of-a-double-dollar-sign-in-bash-makefile
run:
	[ "$$(docker ps -a | grep $(DOCKER_CONTAINER))" ] && docker stop $(DOCKER_CONTAINER) && docker rm $(DOCKER_CONTAINER)
	docker run -d --restart=unless-stopped --name $(DOCKER_CONTAINER) -p 5000:80 $(DOCKER_IMG_NAME)

debug:
	docker run -it $(DOCKER_IMG_NAME) /bin/bash

push:
	docker login https://ghcr.io/Weather-Routing-Research -u $(GH_USER) --password-stdin < $(GH_TOKEN_FILE)
	docker push $(DOCKER_IMG_NAME):latest
