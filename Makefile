GIT_SHA := $(shell git rev-parse --short HEAD)
IMAGE := dmcli
WORKSPACE ?= $(shell pwd)

docker-build:
	@docker build \
		-t $(IMAGE):$(GIT_SHA) \
		-t $(IMAGE):latest \
		.

docker-run:
	@docker run --rm -it \
		-v $(WORKSPACE):/workspace:rw \
		$(IMAGE):latest

