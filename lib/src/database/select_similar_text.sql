SELECT
    vector_distance_cos(e.embedding, ?) AS score,
    e.text_chunk,
    dc.file_name,
    dc.section
FROM embeddings e
LEFT JOIN document_chunks dc ON dc.embedding_id = e.id
ORDER BY score ASC
LIMIT ?;
