SELECT text_chunk
FROM embeddings
ORDER BY vector_distance_cos(embedding, ?)
ASC LIMIT ?;
