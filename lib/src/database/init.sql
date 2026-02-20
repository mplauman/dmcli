CREATE TABLE IF NOT EXISTS text_embeddings (
    text TEXT NOT NULL,
    embedding F32_BLOB({})
);

CREATE INDEX IF NOT EXISTS text_embeddings_embedding_idx ON text_embeddings (libsql_vector_idx(embedding));
