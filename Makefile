qdrant:
	@docker run --rm \
		-p 6333:6333 \
		-p 6334:6334 \
		-v qdrant_storage:/qdrant/storage:z \
		qdrant/qdrant

qdrant-clean:
	@docker volume rm qdrant_storage

