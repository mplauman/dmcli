use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{Repo, RepoType, api::sync::Api};
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
use crate::index::search_result::{SearchResult, Source};
use crate::result::Result;

const MODEL_NAME: &str = "sentence-transformers/all-MiniLM-L6-v2";
const MODEL_REVISION: &str = "refs/pr/21";
const MODEL_DIMS: usize = 384;
const CHUNK_SIZE: usize = 256;
const OVERLAP: usize = 50;
const COLLECTION: &str = "dmcli_chunks";

pub struct DocumentIndex {
    embedding_builder: EmbeddingBuilder,
    /// URL of the Qdrant instance. When `None`, search and indexing fall back
    /// to the local SQLite vector database.
    qdrant_url: Option<String>,
}

impl DocumentIndex {
    /// Create a new [`DocumentIndex`], optionally backed by a persistent
    /// database at `db_path`. When `qdrant_url` is supplied the instance uses
    /// Qdrant for both indexing and hybrid retrieval; otherwise it falls back
    /// to the local SQLite cosine-similarity search.
    pub async fn new(qdrant_url: Option<&str>) -> Result<Self> {
        let device = Device::Cpu;

        let repo = Repo::with_revision(
            MODEL_NAME.to_string(),
            RepoType::Model,
            MODEL_REVISION.to_string(),
        );
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = api.get("model.safetensors")?;
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
            qdrant_url: qdrant_url.map(str::to_owned),
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

    // -------------------------------------------------------------------------
    // Qdrant helpers
    // -------------------------------------------------------------------------

    /// Connect to Qdrant and ensure the `dmcli_chunks` collection exists with
    /// the correct named-vector schema:
    ///
    /// - `"dense"` — 384-dim Cosine HNSW vectors
    /// - `"sparse"` — sparse vectors (log-TF token weights)
    async fn qdrant_client(&self, url: &str) -> Result<qdrant_client::Qdrant> {
        use qdrant_client::{
            Qdrant,
            qdrant::{
                CreateCollectionBuilder, Distance, SparseVectorConfig, SparseVectorParams,
                VectorParamsBuilder, VectorParamsMap, VectorsConfig, vectors_config,
            },
        };

        let client = Qdrant::from_url(url)
            .build()
            .map_err(|e| Error::Index(format!("Failed to connect to Qdrant at {url}: {e}")))?;

        if !client
            .collection_exists(COLLECTION)
            .await
            .map_err(|e| Error::Index(format!("Qdrant collection_exists failed: {e}")))?
        {
            // Named dense vectors.
            let dense_params =
                VectorParamsBuilder::new(MODEL_DIMS as u64, Distance::Cosine).build();
            let vectors_config = VectorsConfig {
                config: Some(vectors_config::Config::ParamsMap(VectorParamsMap {
                    map: [("dense".to_string(), dense_params)].into(),
                })),
            };

            // Named sparse vectors.
            let sparse_config = SparseVectorConfig {
                map: [("sparse".to_string(), SparseVectorParams::default())].into(),
            };

            client
                .create_collection(
                    CreateCollectionBuilder::new(COLLECTION)
                        .vectors_config(vectors_config)
                        .sparse_vectors_config(sparse_config),
                )
                .await
                .map_err(|e| Error::Index(format!("Failed to create collection: {e}")))?;
        }

        Ok(client)
    }

    /// Upsert a batch of chunks into Qdrant. Each point carries:
    /// - `"dense"` vector (pooled + normalised BERT embedding)
    /// - `"sparse"` vector (log-TF token weights)
    /// - payload: `content`, `path`, `section`, `hash`
    async fn qdrant_upsert(
        &self,
        url: &str,
        chunks: &[(&str, PathBuf, String, String)],
    ) -> Result<()> {
        use qdrant_client::qdrant::{PointStruct, UpsertPointsBuilder, Vector};

        if chunks.is_empty() {
            return Ok(());
        }

        let client = self.qdrant_client(url).await?;

        let texts: Vec<&str> = chunks.iter().map(|(t, ..)| *t).collect();
        let dense_vecs = self.embedding_builder.encode_dense(texts)?;

        let mut points: Vec<PointStruct> = Vec::with_capacity(chunks.len());
        for (i, (text, path, hash, section)) in chunks.iter().enumerate() {
            let (sparse_indices, sparse_values) = self.embedding_builder.sparse_vector(text)?;

            let mut vectors: HashMap<String, Vector> = HashMap::new();
            vectors.insert("dense".to_string(), dense_vecs[i].to_vec().into());
            if !sparse_indices.is_empty() {
                vectors.insert(
                    "sparse".to_string(),
                    Vector::new_sparse(sparse_indices, sparse_values),
                );
            }

            let mut payload = qdrant_client::Payload::new();
            payload.insert("content", text.to_string());
            payload.insert("path", path.to_string_lossy().as_ref());
            payload.insert("section", section.as_str());
            payload.insert("hash", hash.as_str());

            // Derive a deterministic u64 point ID from the content hash.
            let id = u64::from_be_bytes(
                Sha256::digest(hash.as_bytes())[..8]
                    .try_into()
                    .expect("slice is 8 bytes"),
            );

            points.push(PointStruct::new(id, vectors, payload));
        }

        client
            .upsert_points(UpsertPointsBuilder::new(COLLECTION, points).wait(true))
            .await
            .map_err(|e| Error::Index(format!("Qdrant upsert failed: {e}")))?;

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /// Search for documents relevant to `text`.
    ///
    /// When a Qdrant URL is configured, executes a hybrid query:
    /// - **Dense prefetch**: top-`2×MAX_RESULTS` nearest neighbours by BERT embedding
    /// - **Sparse prefetch**: top-`2×MAX_RESULTS` by log-TF keyword weights derived
    ///   from the semantically salient keywords in the query
    ///
    /// Both candidate sets are fused with Reciprocal Rank Fusion (RRF).
    /// Falls back to local SQLite cosine-similarity search when no URL is set.
    pub async fn search(&self, text: &str, max_results: u64) -> Result<Vec<SearchResult>> {
        let Some(url) = &self.qdrant_url else {
            return Ok(vec![]);
        };

        use qdrant_client::qdrant::{
            PrefetchQueryBuilder, Query, QueryPointsBuilder, RrfBuilder, SearchParamsBuilder,
            VectorInput,
        };

        let prefetch_limit: u64 = max_results * 2;
        // RRF scores rank 1 at ~0.016 (1/(k+1), k=60). 0.010 filters results
        // that ranked poorly in both lists. For pure dense, 0.5 is a
        // reasonable cosine similarity floor for "meaningfully related".
        const RRF_SCORE_THRESHOLD: f32 = 0.010;
        const DENSE_SCORE_THRESHOLD: f32 = 0.50;

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
            .map(|a| a.to_vec())
            .unwrap_or_default();

        // Sparse query vector: extract keywords from the query and map each to
        // token IDs, accumulating scores across phrases.
        let keywords = self.embedding_builder.extract_keywords(text, 3, 6, 0.45)?;
        println!("Keywords: {keywords:?}");

        let mut sparse_map: HashMap<u32, f32> = HashMap::new();
        for (phrase, score) in &keywords {
            for word in phrase.split_whitespace() {
                if let Some(id) = self.embedding_builder.tokenizer.token_to_id(word) {
                    *sparse_map.entry(id).or_default() += score;
                }
            }
        }
        let (sparse_indices, sparse_values): (Vec<u32>, Vec<f32>) = sparse_map.into_iter().unzip();

        let client = self.qdrant_client(&url).await?;

        let dense_prefetch = PrefetchQueryBuilder::default()
            .query(Query::new_nearest(VectorInput::new_dense(dense_vec)))
            .using("dense")
            .limit(prefetch_limit)
            .params(SearchParamsBuilder::default().hnsw_ef(128).exact(false));

        let mut query_builder = QueryPointsBuilder::new(COLLECTION)
            .add_prefetch(dense_prefetch)
            .with_payload(true)
            .limit(max_results);

        if !sparse_indices.is_empty() {
            let sparse_prefetch = PrefetchQueryBuilder::default()
                .query(Query::new_nearest(VectorInput::new_sparse(
                    sparse_indices,
                    sparse_values,
                )))
                .using("sparse")
                .limit(prefetch_limit);

            query_builder = query_builder
                .add_prefetch(sparse_prefetch)
                .query(Query::new_rrf(RrfBuilder::new().weights(vec![1.0, 1.0])))
                .score_threshold(RRF_SCORE_THRESHOLD);
        } else {
            // No usable keywords — pure dense.
            query_builder = query_builder
                .query(Query::new_nearest(VectorInput::new_dense(
                    self.embedding_builder
                        .encode_dense(vec![query_chunk.as_str()])?
                        .into_iter()
                        .next()
                        .unwrap_or_default(),
                )))
                .using("dense")
                .score_threshold(DENSE_SCORE_THRESHOLD);
        }

        let response = client
            .query(query_builder.build())
            .await
            .map_err(|e| Error::Index(format!("Qdrant query failed: {e}")))?;

        let results = response
            .result
            .into_iter()
            .map(|point| {
                let score = point.score;
                let payload = point.payload;
                let text = payload
                    .get("content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.as_str())
                    .unwrap_or("")
                    .to_string();
                let source = match (payload.get("path"), payload.get("section")) {
                    (Some(p), Some(s)) => {
                        let path = p.as_str().map(|s| s.as_str()).unwrap_or("");
                        let section = s.as_str().map(|s| s.as_str()).unwrap_or("").to_string();
                        Some(Source::File {
                            path: PathBuf::from(path),
                            section,
                        })
                    }
                    _ => None,
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

    /// Index all Markdown files under `path`.
    pub async fn index_path(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let chunk_config = ChunkConfig::new(CHUNK_SIZE)
            .with_sizer(&self.embedding_builder.tokenizer)
            .with_overlap(OVERLAP)
            .expect("Overlap is a sane value");
        let splitter = MarkdownSplitter::new(chunk_config);

        let walker = WalkDir::new(path)
            .into_iter()
            .filter_entry(|e| {
                !e.file_name()
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

            if let Some(url) = &self.qdrant_url.clone() {
                let batch: Vec<(&str, PathBuf, String, String)> = sub_chunks
                    .iter()
                    .map(|(t, p, h, s)| (t.as_str(), p.clone(), h.clone(), s.clone()))
                    .collect();
                self.qdrant_upsert(&url, &batch).await?;
            }
        }

        Ok(())
    }

    /// Index a raw string.
    pub async fn index_str(&mut self, text: &str) -> Result<()> {
        if let Some(url) = &self.qdrant_url.clone() {
            let chunks = self.split_text(text)?;
            let batch: Vec<(&str, PathBuf, String, String)> = chunks
                .iter()
                .map(|&chunk| {
                    let hash = hex::encode(Sha256::digest(chunk.as_bytes()));
                    (chunk, PathBuf::new(), hash, String::new())
                })
                .collect();
            self.qdrant_upsert(&url, &batch).await?;
        }
        Ok(())
    }
}
