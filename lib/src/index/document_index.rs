use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{Repo, RepoType, api::tokio::Api};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::read_to_string;
use std::path::{Path, PathBuf};
use text_splitter::{ChunkConfig, MarkdownSplitter, TextSplitter};
use tokenizers::{PaddingParams, Tokenizer};
use walkdir::{DirEntry, WalkDir};

use crate::error::Error;
use crate::index::embedding::EmbeddingBuilder;
use crate::index::markdown::process_markdown;
use crate::index::search_result::SearchResult;
use crate::index::vector_store::{Chunk, VectorStore};
use crate::result::Result;

const MODEL_NAME: &str = "sentence-transformers/all-MiniLM-L6-v2";
const MODEL_REVISION: &str = "refs/pr/21";
const CHUNK_SIZE: usize = 256;
const OVERLAP: usize = 50;

/// Drives document chunking, embedding, and retrieval against a [`VectorStore`].
pub struct DocumentIndex {
    embedding_builder: EmbeddingBuilder,
    store: Box<dyn VectorStore>,
}

impl DocumentIndex {
    /// Create a new [`DocumentIndex`] backed by the given [`VectorStore`].
    ///
    /// Downloads the BERT model weights on first use (cached by `hf-hub`).
    pub async fn new(store: Box<dyn VectorStore>) -> Result<Self> {
        let device = Device::Cpu;

        let repo = Repo::with_revision(
            MODEL_NAME.to_string(),
            RepoType::Model,
            MODEL_REVISION.to_string(),
        );
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json").await?;
            let tokenizer = api.get("tokenizer.json").await?;
            let weights = api.get("model.safetensors").await?;
            (config, tokenizer, weights)
        };

        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| Error::Index(format!("Failed to initialize tokenizer: {e:?}")))?;

        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest;
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            embedding_builder: EmbeddingBuilder {
                device,
                tokenizer,
                model,
            },
            store,
        })
    }

    fn split_text<'a>(&self, text: &'a str) -> Result<Vec<&'a str>> {
        let chunk_config = ChunkConfig::new(CHUNK_SIZE)
            .with_sizer(&self.embedding_builder.tokenizer)
            .with_overlap(CHUNK_SIZE / 4)
            .expect("Overlap is a sane value");

        let splitter = TextSplitter::new(chunk_config);
        Ok(splitter.chunks(text).collect())
    }

    /// Embed `text_chunks` and return them as [`Chunk`]s ready for the store.
    fn embed_chunks(&self, text_chunks: &[(&str, PathBuf, String, String)]) -> Result<Vec<Chunk>> {
        if text_chunks.is_empty() {
            return Ok(vec![]);
        }

        let texts: Vec<&str> = text_chunks.iter().map(|(t, ..)| *t).collect();
        let dense_vecs = self.embedding_builder.encode_dense(texts)?;

        let mut chunks = Vec::with_capacity(text_chunks.len());
        for (i, (text, path, hash, section)) in text_chunks.iter().enumerate() {
            let (sparse_indices, sparse_values) = self.embedding_builder.sparse_vector(text)?;
            chunks.push(Chunk {
                text: text.to_string(),
                path: path.clone(),
                hash: hash.clone(),
                section: section.clone(),
                dense: dense_vecs[i].clone(),
                sparse_indices,
                sparse_values,
            });
        }

        Ok(chunks)
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /// Search for documents relevant to `text`.
    ///
    /// Executes a hybrid query against the configured store:
    /// - **Dense**: top-`2×max_results` nearest neighbours by BERT embedding
    /// - **Sparse**: top-`2×max_results` by log-TF keyword weights derived
    ///   from the semantically salient keywords in the query
    ///
    /// Both candidate sets are fused with Reciprocal Rank Fusion (RRF) when
    /// the store supports it.
    pub async fn search(&self, text: &str, max_results: u64) -> Result<Vec<SearchResult>> {
        // Dense query vector.
        let query_chunk = self
            .split_text(text)?
            .into_iter()
            .next()
            .map(|s| s.to_string())
            .unwrap_or_else(|| text.to_string());

        let dense_vec: Vec<f32> = self
            .embedding_builder
            .encode_dense(vec![query_chunk.as_str()])?
            .into_iter()
            .next()
            .unwrap_or_default();

        // Sparse query vector: extract keywords from the query and map each to
        // token IDs, accumulating scores across phrases.
        let keywords = self.embedding_builder.extract_keywords(text, 3, 6, 0.45)?;

        let mut sparse_map: HashMap<u32, f32> = HashMap::new();
        for (phrase, score) in &keywords {
            for word in phrase.split_whitespace() {
                if let Some(id) = self.embedding_builder.tokenizer.token_to_id(word) {
                    *sparse_map.entry(id).or_default() += score;
                }
            }
        }
        let (sparse_indices, sparse_values): (Vec<u32>, Vec<f32>) = sparse_map.into_iter().unzip();

        self.store
            .search(dense_vec, sparse_indices, sparse_values, max_results)
            .await
    }

    /// Index all Markdown files under `path`.
    pub async fn index_path(&self, path: impl AsRef<Path>) -> Result<()> {
        let chunk_config = ChunkConfig::new(CHUNK_SIZE)
            .with_sizer(&self.embedding_builder.tokenizer)
            .with_overlap(OVERLAP)
            .expect("Overlap is a sane value");
        let splitter = MarkdownSplitter::new(chunk_config);

        let walker = WalkDir::new(path)
            .into_iter()
            .filter_entry(|e| {
                // Always descend into the root (depth 0); only skip hidden
                // entries (names starting with '.') inside it.
                e.depth() == 0
                    || !e
                        .file_name()
                        .to_str()
                        .map(|name| name.starts_with('.'))
                        .unwrap_or(false)
            })
            .filter_map(|e| e.map(DirEntry::into_path).ok())
            .filter(|e| e.is_file())
            .filter(|e| e.extension().and_then(|e| e.to_str()) == Some("md"));

        for entry in walker {
            let contents = read_to_string(&entry)?;
            let chunks = process_markdown(&contents, entry.as_path());

            // (text, path, hash, section)
            let sub_chunks: Vec<(String, PathBuf, String, String)> = chunks
                .into_iter()
                .flat_map(|chunk| {
                    splitter
                        .chunks(&chunk.content)
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .map(|s| {
                            let hash = hex::encode(Sha256::digest(s.as_bytes()));
                            let section = chunk.headers.join("/");
                            (s, chunk.path.to_path_buf(), hash, section)
                        })
                        .collect::<Vec<_>>()
                })
                .collect();

            let batch: Vec<(&str, PathBuf, String, String)> = sub_chunks
                .iter()
                .map(|(t, p, h, s)| (t.as_str(), p.clone(), h.clone(), s.clone()))
                .collect();

            let embedded = self.embed_chunks(&batch)?;
            self.store.upsert(embedded).await?;
        }

        Ok(())
    }

    /// Index a raw string.
    pub async fn index_str(&self, text: &str) -> Result<()> {
        let chunks = self.split_text(text)?;
        let batch: Vec<(&str, PathBuf, String, String)> = chunks
            .iter()
            .map(|&chunk| {
                let hash = hex::encode(Sha256::digest(chunk.as_bytes()));
                (chunk, PathBuf::new(), hash, String::new())
            })
            .collect();

        let embedded = self.embed_chunks(&batch)?;
        self.store.upsert(embedded).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::vector_store::InMemoryStore;
    use std::sync::Arc;

    /// Wrap an [`InMemoryStore`] in an `Arc` so we can inspect it after
    /// handing ownership to [`DocumentIndex`].
    struct SharedInMemoryStore(Arc<InMemoryStore>);

    impl SharedInMemoryStore {
        fn new() -> (Self, Arc<InMemoryStore>) {
            let inner = Arc::new(InMemoryStore::new());
            (Self(Arc::clone(&inner)), inner)
        }
    }

    #[async_trait::async_trait]
    impl VectorStore for SharedInMemoryStore {
        async fn upsert(&self, chunks: Vec<Chunk>) -> Result<()> {
            self.0.upsert(chunks).await
        }

        async fn search(
            &self,
            dense: Vec<f32>,
            sparse_indices: Vec<u32>,
            sparse_values: Vec<f32>,
            max_results: u64,
        ) -> Result<Vec<SearchResult>> {
            self.0
                .search(dense, sparse_indices, sparse_values, max_results)
                .await
        }
    }

    /// Build a [`DocumentIndex`] backed by an [`InMemoryStore`] and return
    /// both the index and a shared handle to the store for inspection.
    async fn make_index() -> (DocumentIndex, Arc<InMemoryStore>) {
        let (wrapper, store_ref) = SharedInMemoryStore::new();
        let index = DocumentIndex::new(Box::new(wrapper))
            .await
            .expect("Failed to build DocumentIndex");
        (index, store_ref)
    }

    #[tokio::test]
    async fn index_str_stores_chunks() {
        let (index, store) = make_index().await;
        index
            .index_str("The quick brown fox jumps over the lazy dog.")
            .await
            .expect("index_str failed");

        assert!(
            !store.stored_chunks().is_empty(),
            "store should contain at least one chunk after indexing"
        );
    }

    #[tokio::test]
    async fn index_str_chunk_text_matches_input() {
        let (index, store) = make_index().await;
        let text = "Paladins swear an Oath of Devotion to fight evil.";
        index.index_str(text).await.expect("index_str failed");

        let chunks = store.stored_chunks();
        assert!(
            chunks.iter().any(|c| c.text.contains("Paladins")),
            "stored chunk should contain the original text"
        );
    }

    #[tokio::test]
    async fn index_str_embeddings_are_non_zero() {
        let (index, store) = make_index().await;
        index
            .index_str("A wizard casts fireball.")
            .await
            .expect("index_str failed");

        for chunk in store.stored_chunks() {
            let norm: f32 = chunk.dense.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(norm > 0.0, "dense embedding should not be the zero vector");
        }
    }

    #[tokio::test]
    async fn search_returns_relevant_result() {
        let (index, _store) = make_index().await;

        index
            .index_str("Druids can wildshape into animals.")
            .await
            .expect("index_str failed");

        let results = index
            .search("wildshape transformation", 3)
            .await
            .expect("search failed");

        assert!(
            !results.is_empty(),
            "search should return at least one result"
        );
    }

    #[tokio::test]
    async fn search_scores_are_bounded() {
        let (index, _store) = make_index().await;

        index
            .index_str("Rangers are skilled trackers and hunters.")
            .await
            .expect("index_str failed");

        let results = index
            .search("tracking and hunting", 5)
            .await
            .expect("search failed");

        for result in &results {
            assert!(
                result.score >= -1.0 && result.score <= 1.0,
                "cosine similarity must be in [-1, 1], got {}",
                result.score
            );
        }
    }

    #[tokio::test]
    async fn index_str_is_idempotent() {
        let (index, store) = make_index().await;
        let text = "Rogues excel at stealth and deception.";

        index.index_str(text).await.expect("first index_str failed");
        let count_after_first = store.stored_chunks().len();

        index
            .index_str(text)
            .await
            .expect("second index_str failed");
        let count_after_second = store.stored_chunks().len();

        assert_eq!(
            count_after_first, count_after_second,
            "re-indexing identical text should not grow the store (upsert semantics)"
        );
    }

    #[tokio::test]
    async fn index_path_indexes_markdown_files() {
        let (index, store) = make_index().await;

        let dir = tempfile::tempdir().expect("failed to create temp dir");
        std::fs::write(
            dir.path().join("spells.md"),
            "# Fireball\nA bright streak flashes from your finger.",
        )
        .expect("failed to write temp file");

        index
            .index_path(dir.path())
            .await
            .expect("index_path failed");

        let chunks = store.stored_chunks();
        assert!(
            !chunks.is_empty(),
            "store should contain chunks after indexing a directory"
        );
        assert!(
            chunks
                .iter()
                .any(|c| c.text.contains("Fireball") || c.text.contains("bright streak")),
            "chunks should contain content from the markdown file"
        );
    }

    #[tokio::test]
    async fn index_path_ignores_non_markdown_files() {
        let (index, store) = make_index().await;

        let dir = tempfile::tempdir().expect("failed to create temp dir");
        std::fs::write(dir.path().join("notes.txt"), "This is a plain text file.")
            .expect("failed to write temp file");

        index
            .index_path(dir.path())
            .await
            .expect("index_path failed");

        assert!(
            store.stored_chunks().is_empty(),
            "non-markdown files should not be indexed"
        );
    }
}
