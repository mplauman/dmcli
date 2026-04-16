GIT_SHA := $(shell git rev-parse --short HEAD)
IMAGE := dmcli
WORKSPACE ?= $(shell pwd)
UID := $(shell id -u)
GID := $(shell id -g)

docker-build:
	@docker build \
		-t $(IMAGE):$(GIT_SHA) \
		-t $(IMAGE):latest \
		.

docker-run:
	@docker run --rm -it \
		--user $(UID):$(GID) \
		-w /workspace \
		-v $(WORKSPACE):/workspace:rw \
		-v dmcli-home:/home/user \
		$(IMAGE):latest

docker-clean:
	@docker volume rm dmcli-home
