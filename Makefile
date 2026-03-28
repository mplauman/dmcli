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
		-v $(WORKSPACE):/workspace:ro \
		$(IMAGE):latest

qdrant:
	@docker run --rm \
		-p 6333:6333 \
		-p 6334:6334 \
		-v qdrant_storage:/qdrant/storage:z \
		qdrant/qdrant

qdrant-clean:
	@docker volume rm qdrant_storage

