use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{Repo, RepoType, api::sync::Api};
use pulldown_cmark::{Event, HeadingLevel, Parser, Tag, TagEnd};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::read_to_string;
use std::path::{Path, PathBuf};
use text_splitter::{ChunkConfig, MarkdownSplitter, TextSplitter};
use tokenizers::{EncodeInput, PaddingParams, Tokenizer};
use walkdir::{DirEntry, WalkDir};

use crate::database::SearchResult;
use crate::error::Error;
use crate::result::Result;

const MODEL_NAME: &str = "sentence-transformers/all-MiniLM-L6-v2";
const MODEL_REVISION: &str = "refs/pr/21";
const MODEL_DIMS: usize = 384;
const CHUNK_SIZE: usize = 256;
const OVERLAP: usize = 50;
const COLLECTION: &str = "dmcli_chunks";

#[derive(Debug, Clone)]
struct ChunkPayload<'a> {
    pub content: String,
    pub path: &'a Path,
    pub headers: Vec<String>,
}

struct EmbeddingBuilder {
    device: Device,
    tokenizer: Tokenizer,
    model: BertModel,
}

impl EmbeddingBuilder {
    /// Extract salient keywords from `prompt` by comparing n-gram candidates
    /// against the global intent vector and returning the top-scoring,
    /// non-redundant phrases above `threshold`.
    fn extract_keywords(
        &self,
        prompt: &str,
        max_n: usize,
        top_k: usize,
        threshold: f32,
    ) -> Result<Vec<(String, f32)>> {
        let stop_words = include_str!("stopwords.txt")
            .split_whitespace()
            .collect::<Vec<_>>();

        let words = prompt
            .to_lowercase()
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !stop_words.contains(w) && w.len() > 2)
            .map(|w| w.to_string())
            .collect::<Vec<_>>();

        let mut candidates = Vec::new();
        for n in 1..=max_n {
            for window in words.windows(n) {
                candidates.push(window.join(" "));
            }
        }

        if candidates.is_empty() {
            return Ok(vec![]);
        }
        if candidates.len() == 1 {
            return Ok(vec![(candidates.pop().unwrap(), 1.0)]);
        }

        // Single vector representing the global intent of the prompt.
        let global_vec = self.encode(vec![prompt])?;
        let global_vec = global_vec.pool_2()?;
        let global_vec = global_vec.normalize()?;
        let global_vec = global_vec.embedding;

        let pooled_words = self.encode(candidates.clone())?;
        let pooled_words = pooled_words.pool_2()?;
        let pooled_words = pooled_words.normalize()?;
        let pooled_words = pooled_words.embedding;

        // [N, HiddenSize] * [HiddenSize, 1] -> [N]
        let similarities = pooled_words
            .matmul(&global_vec.unsqueeze(1)?)?
            .flatten_all()?;
        let scores: Vec<f32> = similarities.to_vec1()?;

        let mut results = candidates.into_iter().zip(scores).collect::<Vec<_>>();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.dedup_by(|a, b| a.0 == b.0);

        let results = results
            .into_iter()
            .filter(|(_, score)| *score >= threshold)
            .collect::<Vec<_>>();

        let mut final_keywords: Vec<(String, f32)> = Vec::new();
        for (phrase, score) in results {
            let is_redundant = final_keywords.iter().any(|(existing, _)| {
                existing.contains(&phrase) || phrase.contains(existing.as_str())
            });
            if !is_redundant {
                final_keywords.push((phrase, score));
            }
            if final_keywords.len() >= top_k {
                break;
            }
        }

        Ok(final_keywords)
    }

    /// Build a sparse vector from `text` by tokenizing it and computing
    /// log-normalised term frequencies. Special tokens are excluded by passing
    /// `false` to `add_special_tokens`. Returns parallel `(indices, values)` vecs.
    fn sparse_vector(&self, text: &str) -> Result<(Vec<u32>, Vec<f32>)> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| Error::Index(format!("Tokenizer error: {e}")))?;

        let mut counts: HashMap<u32, u32> = HashMap::new();
        for &id in encoding.get_ids() {
            *counts.entry(id).or_default() += 1;
        }

        if counts.is_empty() {
            return Ok((vec![], vec![]));
        }

        // Log-normalise: weight = 1 + ln(tf)
        let (indices, values): (Vec<u32>, Vec<f32>) = counts
            .into_iter()
            .map(|(id, tf)| (id, 1.0 + (tf as f32).ln()))
            .unzip();

        Ok((indices, values))
    }

    fn encode<'a, E>(&self, input: Vec<E>) -> Result<Embedding>
    where
        E: Into<EncodeInput<'a>> + Send,
    {
        let len = input.len();
        if len == 0 {
            return Err(Error::Index("nothing to encode".to_string()));
        }

        let tokens = self
            .tokenizer
            .encode_batch(input, true)
            .map_err(|e| Error::Index(format!("Failed to encode batch: {e:?}")))?;

        let token_ids = tokens
            .iter()
            .map(|t| {
                let ids = t.get_ids().to_vec();
                Ok(Tensor::new(ids.as_slice(), &self.device)?)
            })
            .collect::<Result<Vec<_>>>()?;

        let attention_mask = tokens
            .iter()
            .map(|t| {
                let mask = t.get_attention_mask().to_vec();
                Ok(Tensor::new(mask.as_slice(), &self.device)?)
            })
            .collect::<Result<Vec<_>>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;
        let token_type_ids = token_ids.zeros_like()?;

        let embeddings = self
            .model
            .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

        Ok(Embedding {
            embedding: embeddings,
            attention_mask,
        })
    }
}

struct Embedding {
    embedding: Tensor,
    attention_mask: Tensor,
}

impl Embedding {
    fn normalize(self) -> Result<Self> {
        let norm = self
            .embedding
            .sqr()?
            .sum_keepdim(self.embedding.rank() - 1)?
            .sqrt()?;
        let embedding = self.embedding.broadcast_div(&norm)?;
        Ok(Self {
            embedding,
            attention_mask: self.attention_mask,
        })
    }

    /// Average-pool over tokens weighted by the attention mask. Produces the
    /// same result as `sentence_transformers` Python library.
    fn pool_1(self) -> Result<Embedding> {
        let mask = self.attention_mask.to_dtype(DTYPE)?.unsqueeze(2)?;
        let sum_mask = mask.sum(1)?;
        let embeddings = (self.embedding.broadcast_mul(&mask)?).sum(1)?;
        let embeddings = embeddings.broadcast_div(&sum_mask)?;
        Ok(Embedding {
            embedding: embeddings,
            attention_mask: self.attention_mask,
        })
    }

    /// Simple mean pool to a single vector.
    fn pool_2(self) -> Result<Embedding> {
        let (_, s, _) = self.embedding.dims3()?;
        let embedding = (self.embedding.sum(1)? / (s as f64))?.squeeze(0)?;
        Ok(Embedding {
            embedding,
            attention_mask: self.attention_mask,
        })
    }
}

fn process_markdown<'a>(content: &'a str, path: &'a Path) -> Vec<ChunkPayload<'a>> {
    let options = pulldown_cmark::Options::ENABLE_TABLES;
    let parser = Parser::new_ext(content, options);
    let mut chunks = Vec::new();
    let mut header_stack: Vec<String> = Vec::new();
    let mut current_text = String::new();
    let mut in_header = false;
    let mut current_header_text = String::new();
    let mut current_header_level: usize = 0;

    for event in parser {
        match event {
            Event::Start(Tag::Heading { level, .. }) => {
                // Emit accumulated text as a chunk before starting a new heading.
                let trimmed = current_text.trim();
                if !trimmed.is_empty() {
                    let normalized = trimmed.split_whitespace().collect::<Vec<&str>>().join(" ");
                    chunks.push(ChunkPayload {
                        content: normalized,
                        path,
                        headers: header_stack.clone(),
                    });
                    current_text.clear();
                }
                in_header = true;
                current_header_level = match level {
                    HeadingLevel::H1 => 1,
                    HeadingLevel::H2 => 2,
                    HeadingLevel::H3 => 3,
                    HeadingLevel::H4 => 4,
                    HeadingLevel::H5 => 5,
                    HeadingLevel::H6 => 6,
                };
                current_header_text.clear();
            }
            Event::End(TagEnd::Heading(_)) => {
                in_header = false;
                let target_depth = current_header_level.saturating_sub(1);
                while header_stack.len() > target_depth {
                    header_stack.pop();
                }
                header_stack.push(current_header_text.trim().to_string());
            }
            Event::Text(text) => match in_header {
                true => current_header_text.push_str(&text),
                false => current_text.push_str(&text),
            },
            Event::Code(text) => match in_header {
                true => current_header_text.push_str(&text),
                false => current_text.push_str(&text),
            },
            Event::SoftBreak | Event::HardBreak => match in_header {
                true => current_header_text.push(' '),
                false => current_text.push('\n'),
            },
            _ => {}
        }
    }

    // Emit the final accumulated chunk.
    let trimmed = current_text.trim();
    if !trimmed.is_empty() {
        let normalized = trimmed.split_whitespace().collect::<Vec<&str>>().join(" ");
        chunks.push(ChunkPayload {
            content: normalized,
            path,
            headers: header_stack,
        });
    }

    chunks
}

pub struct DocumentIndex {
    db: crate::database::Database<MODEL_DIMS>,
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
    pub async fn new(db_path: Option<PathBuf>, qdrant_url: Option<&str>) -> Result<Self> {
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

        let db = match db_path {
            Some(path) => crate::database::Database::<MODEL_DIMS>::open(path).await?,
            None => crate::database::Database::<MODEL_DIMS>::new().await?,
        };

        Ok(Self {
            db,
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

    fn encode_dense<'a, E>(&self, input: Vec<E>) -> Result<Vec<[f32; MODEL_DIMS]>>
    where
        E: Into<EncodeInput<'a>> + Send,
    {
        let len = input.len();
        if len == 0 {
            return Ok(vec![]);
        }

        let embedding = self.embedding_builder.encode(input)?;
        let embedding = embedding.pool_1()?;
        let embedding = embedding.normalize()?;

        let mut result: Vec<[f32; MODEL_DIMS]> = Vec::with_capacity(len);
        for i in 0..len {
            result.push(
                embedding
                    .embedding
                    .get(i)
                    .unwrap()
                    .to_vec1::<f32>()?
                    .try_into()
                    .unwrap(),
            );
        }
        Ok(result)
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
        let dense_vecs = self.encode_dense(texts)?;

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
    // Local SQLite helpers
    // -------------------------------------------------------------------------

    async fn local_insert(&mut self, texts: Vec<&str>) -> Result<()> {
        let encodings = self.encode_dense(texts.clone())?;
        for (encoding, text) in encodings.into_iter().zip(texts.into_iter()) {
            self.db.insert(&encoding, text).await?;
        }
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
    pub async fn search<const MAX_RESULTS: u64>(&self, text: &str) -> Result<Vec<SearchResult>> {
        match &self.qdrant_url {
            Some(url) => self.search_qdrant::<MAX_RESULTS>(text, url.clone()).await,
            None => self.search_local::<MAX_RESULTS>(text).await,
        }
    }

    async fn search_qdrant<const MAX_RESULTS: u64>(
        &self,
        text: &str,
        url: String,
    ) -> Result<Vec<SearchResult>> {
        use qdrant_client::qdrant::{
            PrefetchQueryBuilder, Query, QueryPointsBuilder, RrfBuilder, SearchParamsBuilder,
            VectorInput,
        };

        let prefetch_limit: u64 = MAX_RESULTS * 2;
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
            .limit(MAX_RESULTS);

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
                    self.encode_dense(vec![query_chunk.as_str()])?
                        .into_iter()
                        .next()
                        .map(|a| a.to_vec())
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
                        Some(crate::database::Source::File {
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

    async fn search_local<const MAX_RESULTS: u64>(&self, text: &str) -> Result<Vec<SearchResult>> {
        const SCORE_THRESHOLD: f32 = 0.50;

        let chunks = self.split_text(text)?;
        let encodings = self.encode_dense(chunks)?;

        let mut results = Vec::new();
        for encoding in encodings {
            let found = self.db.find_similar(&encoding, MAX_RESULTS).await?;
            results.extend(found);
        }
        results.retain(|r| r.score >= SCORE_THRESHOLD);
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
            } else {
                let texts: Vec<&str> = sub_chunks.iter().map(|(t, ..)| t.as_str()).collect();
                let embeddings = self.encode_dense(texts.clone())?;
                for ((text, path, hash, section), embedding) in
                    sub_chunks.iter().zip(embeddings.into_iter())
                {
                    println!("Inserting {}#{section}", path.display());
                    self.db
                        .insert_doc_chunk(&embedding, text, path, hash, section)
                        .await?;
                }
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
        } else {
            let chunks = self.split_text(text)?;
            self.local_insert(chunks).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn simple_search() {
        let index = DocumentIndex::new(None, None).await.unwrap();

        let sentences = vec![
            "what were the derro doing beneath mog caern in the previous session?",
            "who is naal?",
            "who are naal's friends?",
            "what are the names of the various NPCs in mog caern",
            "who is naal and who are his friends?",
            "what were the derro doing beneath mog caern in the previous adventure?",
            "Hello, world!",
            "Act as a DevOps expert and create a GitHub Action to deploy to AWS Lambda.",
        ];

        for sentence in sentences {
            let results = index.search::<10>(sentence).await.unwrap();
            assert!(results.is_empty());
        }
    }

    #[tokio::test]
    async fn encode_sentences() {
        let mut index = DocumentIndex::new(None, None).await.unwrap();

        let sentences = vec![
            "The cat sits outside",
            "A man is playing guitar",
            "I love pasta",
            "The new movie is awesome",
            "The cat plays in the garden",
            "A woman watches TV",
            "The new movie is so great",
            "Do you like pizza?",
        ];
        for s in sentences {
            index.index_str(s).await.unwrap();
        }

        let results = index.search::<2>("The movie").await.unwrap();
        for result in results {
            println!("{result}");
        }
    }

    #[test]
    fn test_ignore_metadata() {
        let content = r#"---
title: Test
---
"#;
        let chunks = process_markdown(content, Path::new("test.md"));
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn test_process_markdown_simple() {
        let content = r#"---
title: Test
---
# Header 1
Some text.

## Header 2
More text.
"#;
        let chunks = process_markdown(content, Path::new("test.md"));
        assert_eq!(chunks.len(), 2);

        assert_eq!(chunks[0].headers, vec!["Header 1"]);
        assert_eq!(chunks[0].content, "Some text.");

        assert_eq!(chunks[1].headers, vec!["Header 1", "Header 2"]);
        assert_eq!(chunks[1].content, "More text.");
    }

    #[test]
    fn test_process_markdown_complex_hierarchy() {
        let content = r#"
# H1
Text 1

### H3
Text 3

## H2
Text 2
"#;
        let chunks = process_markdown(content, Path::new("test.md"));
        assert_eq!(chunks.len(), 3);

        assert_eq!(chunks[0].headers, vec!["H1"]);
        assert_eq!(chunks[0].content, "Text 1");

        assert_eq!(chunks[1].headers, vec!["H1", "H3"]);
        assert_eq!(chunks[1].content, "Text 3");

        assert_eq!(chunks[2].headers, vec!["H1", "H2"]);
        assert_eq!(chunks[2].content, "Text 2");
    }

    #[test]
    fn test_process_markdown_links_images() {
        let content = r#"
# Links
Here is a [link](https://example.com) and an ![image](img.png).
"#;
        let chunks = process_markdown(content, Path::new("test.md"));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "Here is a link and an image.");
    }

    #[test]
    fn test_hashes_map() {
        let one = process_markdown("hello, world!", Path::new("one.md"));
        let two = process_markdown("hello, world!", Path::new("two.md"));

        for (a, b) in one.into_iter().zip(two.into_iter()) {
            assert_eq!(a.content, b.content);
            assert_eq!(a.headers, b.headers);

            assert_eq!(a.path, Path::new("one.md"));
            assert_eq!(b.path, Path::new("two.md"));
        }
    }
}
