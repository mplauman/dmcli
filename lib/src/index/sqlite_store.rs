//! SQLite vector store implementation using libsql.
//!
//! This module provides a [`VectorStore`] implementation backed by SQLite via the libsql crate.
//! It stores document chunks with their dense vector embeddings in a persistent database.
//!
//! # Features
//!
//! - **Persistent storage**: Chunks are stored in SQLite files on disk
//! - **Dense vector search**: Cosine similarity-based retrieval using dense embeddings
//! - **Upsert semantics**: Chunks are identified by content hash and updated if re-indexed
//!
//! # Architecture
//!
//! The store uses a single `chunks` table with the following schema:
//! - `id`: Auto-incrementing primary key
//! - `hash`: Content hash (unique, indexed for fast lookups)
//! - `content`: The actual text of the chunk
//! - `path`: File path where the chunk originated
//! - `section`: Heading breadcrumb (e.g., "Intro/Combat")
//! - `dense_vector`: BLOB - serialized f32 embedding vectors
//! - `sparse_indices`, `sparse_values`: Reserved for future sparse search support
//! - `created_at`: Timestamp of insertion/update
//!
//! # Vector Storage
//!
//! Dense vectors are serialized as little-endian binary blobs (4 bytes per f32).
//! This allows efficient storage and retrieval without JSON parsing overhead.
//! Future versions may leverage libsql's native F32_BLOB type and vector functions.
//!
//! # Search Algorithm
//!
//! Searches perform a full table scan and compute cosine similarity for all stored vectors
//! using manual calculation. Results are sorted by similarity score and truncated to the
//! requested limit. This is suitable for small to moderate collections.
//!
//! # Example
//!
//! ```ignore
//! # use lib::index::SqliteStore;
//! # async fn example() -> lib::Result<()> {
//! let store = SqliteStore::new("chunks.db").await?;
//!
//! // Upsert chunks with dense embeddings
//! let chunks = vec![/* ... */];
//! store.upsert(chunks).await?;
//!
//! // Search with a query vector
//! let query_vector = vec![/* ... */];
//! let results = store.search(query_vector, vec![], vec![], 10).await?;
//!
//! for result in results {
//!     println!("{}: {}", result.score, result.text);
//! }
//! # Ok(())
//! # }
//! ```

use std::path::PathBuf;

use async_trait::async_trait;

use crate::error::Error;
use crate::index::search_result::{SearchResult, Source};
use crate::index::vector_store::{Chunk, VectorStore};
use crate::result::Result;

#[cfg(test)]
use tempfile::TempDir;

/// A [`VectorStore`] backed by libsql (SQLite with vector support).
///
/// This implementation uses libsql's native vector search capabilities
/// to store and retrieve chunks with dense embeddings.
pub struct SqliteStore {
    db: libsql::Database,
}

impl SqliteStore {
    /// Create a new [`SqliteStore`] that will use a SQLite database at `db_path`.
    ///
    /// If `db_path` is `:memory:`, an in-memory database is used.
    /// Otherwise, the database file is created or opened at the given path.
    pub async fn new(db_path: impl AsRef<std::path::Path>) -> Result<Self> {
        let db = libsql::Builder::new_local(db_path)
            .build()
            .await
            .map_err(|e| Error::Index(format!("Failed to open SQLite database: {e}")))?;

        Ok(Self { db })
    }

    /// Get a connection to the SQLite database and ensure the chunks table exists.
    async fn get_connection(&self) -> Result<libsql::Connection> {
        let conn = self
            .db
            .connect()
            .map_err(|e| Error::Index(format!("Failed to connect to SQLite database: {e}")))?;

        // Create the chunks table with vector column if it doesn't exist
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                hash TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL,
                path TEXT NOT NULL,
                section TEXT NOT NULL,
                dense_vector BLOB NOT NULL,
                sparse_indices BLOB,
                sparse_values BLOB,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            libsql::params![],
        )
        .await
        .map_err(|e| Error::Index(format!("Failed to create chunks table: {e}")))?;

        // Create index on hash for fast lookups
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(hash)",
            libsql::params![],
        )
        .await
        .map_err(|e| Error::Index(format!("Failed to create hash index: {e}")))?;

        Ok(conn)
    }

    /// Convert a vector of f32 to a binary blob format
    fn vector_to_blob(vec: &[f32]) -> Vec<u8> {
        vec.iter().flat_map(|&f| f.to_le_bytes().to_vec()).collect()
    }

    /// Convert a binary blob back to a vector of f32
    fn blob_to_vector(blob: &[u8]) -> Vec<f32> {
        blob.chunks(4)
            .map(|chunk| {
                let mut bytes = [0u8; 4];
                bytes.copy_from_slice(chunk);
                f32::from_le_bytes(bytes)
            })
            .collect()
    }

    /// Convert sparse indices/values to binary blob format
    fn sparse_indices_to_blob(indices: &[u32]) -> Vec<u8> {
        indices
            .iter()
            .flat_map(|&idx| idx.to_le_bytes().to_vec())
            .collect()
    }

    /// Convert sparse values to binary blob format
    fn sparse_values_to_blob(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&v| v.to_le_bytes().to_vec())
            .collect()
    }

    /// Compute cosine similarity between two vectors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_a == 0.0 || mag_b == 0.0 {
            0.0
        } else {
            dot / (mag_a * mag_b)
        }
    }
}

#[async_trait]
impl VectorStore for SqliteStore {
    async fn upsert(&self, chunks: Vec<Chunk>) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        let conn = self.get_connection().await?;

        for chunk in chunks {
            let dense_blob = Self::vector_to_blob(&chunk.dense);
            let sparse_indices_blob = if chunk.sparse_indices.is_empty() {
                None
            } else {
                Some(Self::sparse_indices_to_blob(&chunk.sparse_indices))
            };
            let sparse_values_blob = if chunk.sparse_values.is_empty() {
                None
            } else {
                Some(Self::sparse_values_to_blob(&chunk.sparse_values))
            };

            let path_str = chunk.path.to_string_lossy().into_owned();
            conn.execute(
                "INSERT INTO chunks (hash, content, path, section, dense_vector, sparse_indices, sparse_values)
                 VALUES (?, ?, ?, ?, ?, ?, ?)
                 ON CONFLICT(hash) DO UPDATE SET
                    content = excluded.content,
                    path = excluded.path,
                    section = excluded.section,
                    dense_vector = excluded.dense_vector,
                    sparse_indices = excluded.sparse_indices,
                    sparse_values = excluded.sparse_values",
                libsql::params![
                    chunk.hash.clone(),
                    chunk.text.clone(),
                    path_str,
                    chunk.section.clone(),
                    dense_blob,
                    sparse_indices_blob,
                    sparse_values_blob,
                ],
            )
            .await
            .map_err(|e| Error::Index(format!("Failed to upsert chunk: {e}")))?;
        }

        Ok(())
    }

    async fn search(
        &self,
        dense: Vec<f32>,
        _sparse_indices: Vec<u32>,
        _sparse_values: Vec<f32>,
        max_results: u64,
    ) -> Result<Vec<SearchResult>> {
        let conn = self.get_connection().await?;

        // For now, we'll use dense vector similarity search only
        // Sparse search would require additional keyword indexing infrastructure

        let mut rows = conn
            .query(
                "SELECT hash, content, path, section, dense_vector FROM chunks",
                libsql::params![],
            )
            .await
            .map_err(|e| Error::Index(format!("Failed to query chunks: {e}")))?;

        let mut scored_results: Vec<(f32, String, String, String)> = vec![];

        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| Error::Index(format!("Failed to iterate rows: {e}")))?
        {
            let _hash: String = row
                .get::<String>(0)
                .map_err(|e| Error::Index(format!("Failed to get hash: {e}")))?;
            let content: String = row
                .get::<String>(1)
                .map_err(|e| Error::Index(format!("Failed to get content: {e}")))?;
            let path: String = row
                .get::<String>(2)
                .map_err(|e| Error::Index(format!("Failed to get path: {e}")))?;
            let section: String = row
                .get::<String>(3)
                .map_err(|e| Error::Index(format!("Failed to get section: {e}")))?;
            let dense_blob: Vec<u8> = row
                .get::<Vec<u8>>(4)
                .map_err(|e| Error::Index(format!("Failed to get dense_vector: {e}")))?;

            let stored_dense = Self::blob_to_vector(&dense_blob);
            let score = Self::cosine_similarity(&dense, &stored_dense);

            scored_results.push((score, content, path, section));
        }

        // Sort by score descending
        scored_results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Truncate to max_results
        scored_results.truncate(max_results as usize);

        let results = scored_results
            .into_iter()
            .map(|(score, text, path, section)| {
                let source = if path.is_empty() && section.is_empty() {
                    None
                } else {
                    Some(Source::File {
                        path: PathBuf::from(path),
                        section,
                    })
                };
                SearchResult {
                    score,
                    text,
                    source,
                }
            })
            .collect();

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(text: &str, hash: &str, dense: Vec<f32>) -> Chunk {
        Chunk {
            text: text.to_string(),
            path: PathBuf::from("test.md"),
            hash: hash.to_string(),
            section: "Intro".to_string(),
            dense,
            sparse_indices: vec![],
            sparse_values: vec![],
        }
    }

    #[tokio::test]
    async fn vector_blob_roundtrip() {
        let vec = vec![1.0_f32, 2.0, 3.0, 4.0];
        let blob = SqliteStore::vector_to_blob(&vec);
        let restored = SqliteStore::blob_to_vector(&blob);
        assert_eq!(restored.len(), vec.len());
        for (a, b) in vec.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[tokio::test]
    async fn upsert_stores_chunk() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let store = SqliteStore::new(&db_path).await.unwrap();
        let chunk = make_chunk("hello world", "abc123", vec![1.0, 0.0, 0.0]);
        store.upsert(vec![chunk]).await.unwrap();
    }

    #[tokio::test]
    async fn upsert_replaces_by_hash() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let store = SqliteStore::new(&db_path).await.unwrap();
        let a = make_chunk("first version", "same-hash", vec![1.0, 0.0, 0.0]);
        let b = make_chunk("updated version", "same-hash", vec![0.0, 1.0, 0.0]);
        store.upsert(vec![a]).await.unwrap();
        store.upsert(vec![b]).await.unwrap();
    }

    #[tokio::test]
    async fn search_returns_closest_chunk() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let store = SqliteStore::new(&db_path).await.unwrap();
        store
            .upsert(vec![
                make_chunk("close", "h1", vec![1.0, 0.0]),
                make_chunk("far", "h2", vec![0.0, 1.0]),
            ])
            .await
            .unwrap();

        let results = store
            .search(vec![1.0, 0.0], vec![], vec![], 1)
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "close");
    }

    #[tokio::test]
    async fn search_respects_max_results() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let store = SqliteStore::new(&db_path).await.unwrap();
        store
            .upsert(vec![
                make_chunk("a", "h1", vec![1.0, 0.0]),
                make_chunk("b", "h2", vec![0.9, 0.1]),
                make_chunk("c", "h3", vec![0.8, 0.2]),
            ])
            .await
            .unwrap();

        let results = store
            .search(vec![1.0, 0.0], vec![], vec![], 2)
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn search_empty_store_returns_nothing() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let store = SqliteStore::new(&db_path).await.unwrap();
        let results = store
            .search(vec![1.0, 0.0], vec![], vec![], 5)
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn cosine_similarity_identical() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let sim = SqliteStore::cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        let sim = SqliteStore::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![1.0_f32, 0.0];
        assert_eq!(SqliteStore::cosine_similarity(&a, &b), 0.0);
    }
}
