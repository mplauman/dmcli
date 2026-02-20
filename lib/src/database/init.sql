CREATE TABLE IF NOT EXISTS embeddings(
    id ROWID,
    embedding F32_BLOB({}),
    text_chunk TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS embeddings_embedding_idx ON embeddings (libsql_vector_idx(embedding));

CREATE TABLE IF NOT EXISTS document_chunks(
    embedding_id ROWID,
    file_name TEXT NOT NULL,
    section TEXT NOT NULL,
    hash TEXT NOT NULL,

    FOREIGN KEY(embedding_id) REFERENCES embeddings(id)
);
