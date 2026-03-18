use std::path::PathBuf;

use async_trait::async_trait;

use crate::index::search_result::SearchResult;
use crate::result::Result;

/// A chunk ready to be stored in a vector backend.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Raw text content of the chunk.
    pub text: String,
    /// File path the chunk originated from (empty for raw-string indexing).
    pub path: PathBuf,
    /// SHA-256 hex digest of the chunk text.
    pub hash: String,
    /// Slash-joined heading breadcrumb, e.g. `"Intro/Combat"`.
    pub section: String,
    /// Normalised dense embedding vector.
    pub dense: Vec<f32>,
    /// Sparse token indices (parallel to `sparse_values`).
    pub sparse_indices: Vec<u32>,
    /// Sparse token weights (parallel to `sparse_indices`).
    pub sparse_values: Vec<f32>,
}

/// Abstraction over a vector-capable storage backend.
///
/// Implementors are responsible for persisting and retrieving [`Chunk`]s.
/// The [`DocumentIndex`] drives chunking, embedding, and keyword extraction;
/// the store only needs to handle persistence and similarity retrieval.
///
/// [`DocumentIndex`]: crate::index::DocumentIndex
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Upsert a batch of pre-embedded chunks into the store.
    async fn upsert(&self, chunks: Vec<Chunk>) -> Result<()>;

    /// Return up to `max_results` chunks most relevant to the given query
    /// vectors.
    ///
    /// Both the dense query vector and optional sparse keyword vectors are
    /// provided so the backend can use whichever retrieval strategy it
    /// supports.
    async fn search(
        &self,
        dense: Vec<f32>,
        sparse_indices: Vec<u32>,
        sparse_values: Vec<f32>,
        max_results: u64,
    ) -> Result<Vec<SearchResult>>;
}

// -----------------------------------------------------------------------------
// No-op store (null object / disabled backend)
// -----------------------------------------------------------------------------

/// A [`VectorStore`] that silently discards all writes and returns no results.
///
/// Use this when no vector backend is configured and search/indexing should
/// gracefully do nothing rather than fail.
pub struct NoopStore;

#[async_trait]
impl VectorStore for NoopStore {
    async fn upsert(&self, _chunks: Vec<Chunk>) -> Result<()> {
        Ok(())
    }

    async fn search(
        &self,
        _dense: Vec<f32>,
        _sparse_indices: Vec<u32>,
        _sparse_values: Vec<f32>,
        _max_results: u64,
    ) -> Result<Vec<SearchResult>> {
        Ok(vec![])
    }
}

// -----------------------------------------------------------------------------
// In-memory store (for unit tests)
// -----------------------------------------------------------------------------

/// A trivial in-memory [`VectorStore`] that stores chunks verbatim and
/// performs brute-force cosine similarity search over dense vectors.
///
/// This implementation is intentionally simple and is only intended for use
/// in unit tests.
#[cfg(test)]
pub struct InMemoryStore {
    chunks: std::sync::Mutex<Vec<Chunk>>,
}

#[cfg(test)]
impl InMemoryStore {
    /// Create an empty [`InMemoryStore`].
    pub fn new() -> Self {
        Self {
            chunks: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Return a snapshot of all stored chunks.
    pub fn stored_chunks(&self) -> Vec<Chunk> {
        self.chunks
            .lock()
            .expect("InMemoryStore mutex poisoned")
            .clone()
    }
}

#[cfg(test)]
impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute cosine similarity between two equal-length vectors.
/// Returns 0.0 when either vector has zero magnitude.
#[cfg(test)]
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        0.0
    } else {
        dot / (mag_a * mag_b)
    }
}

#[cfg(test)]
#[async_trait]
impl VectorStore for InMemoryStore {
    async fn upsert(&self, chunks: Vec<Chunk>) -> Result<()> {
        let mut store = self.chunks.lock().expect("InMemoryStore mutex poisoned");
        for chunk in chunks {
            // Replace existing chunk with the same hash (upsert semantics).
            if let Some(pos) = store.iter().position(|c| c.hash == chunk.hash) {
                store[pos] = chunk;
            } else {
                store.push(chunk);
            }
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
        let store = self.chunks.lock().expect("InMemoryStore mutex poisoned");

        let mut scored: Vec<(f32, &Chunk)> = store
            .iter()
            .map(|c| (cosine(&dense, &c.dense), c))
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(max_results as usize);

        let results = scored
            .into_iter()
            .map(|(score, chunk)| {
                let source = if chunk.path == PathBuf::new() && chunk.section.is_empty() {
                    None
                } else {
                    Some(crate::index::search_result::Source::File {
                        path: chunk.path.clone(),
                        section: chunk.section.clone(),
                    })
                };
                SearchResult {
                    score,
                    text: chunk.text.clone(),
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
    async fn upsert_stores_chunks() {
        let store = InMemoryStore::new();
        let chunk = make_chunk("hello world", "abc123", vec![1.0, 0.0, 0.0]);
        store.upsert(vec![chunk]).await.unwrap();
        assert_eq!(store.stored_chunks().len(), 1);
    }

    #[tokio::test]
    async fn upsert_replaces_by_hash() {
        let store = InMemoryStore::new();
        let a = make_chunk("first version", "same-hash", vec![1.0, 0.0, 0.0]);
        let b = make_chunk("updated version", "same-hash", vec![0.0, 1.0, 0.0]);
        store.upsert(vec![a]).await.unwrap();
        store.upsert(vec![b]).await.unwrap();

        let chunks = store.stored_chunks();
        assert_eq!(chunks.len(), 1, "upsert should replace, not append");
        assert_eq!(chunks[0].text, "updated version");
    }

    #[tokio::test]
    async fn search_returns_closest_chunk() {
        let store = InMemoryStore::new();
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
        let store = InMemoryStore::new();
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
        let store = InMemoryStore::new();
        let results = store
            .search(vec![1.0, 0.0], vec![], vec![], 5)
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn cosine_identical_unit_vectors() {
        let a = vec![1.0_f32, 0.0, 0.0];
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        assert!(cosine(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_zero_vector_returns_zero() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![1.0_f32, 0.0];
        assert_eq!(cosine(&a, &b), 0.0);
    }
}
