use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use sha2::{Digest, Sha256};

use crate::error::Error;
use crate::index::search_result::{SearchResult, Source};
use crate::index::vector_store::{Chunk, VectorStore};
use crate::result::Result;

const COLLECTION: &str = "dmcli_chunks";
const MODEL_DIMS: u64 = 384;

/// A [`VectorStore`] backed by a remote Qdrant instance.
///
/// The collection is created automatically on first use with named
/// `"dense"` (384-dim Cosine HNSW) and `"sparse"` vector schemas.
pub struct QdrantStore {
    url: String,
}

impl QdrantStore {
    /// Create a new [`QdrantStore`] that will connect to the Qdrant instance
    /// at `url`.
    pub fn new(url: impl Into<String>) -> Self {
        Self { url: url.into() }
    }

    /// Connect to Qdrant and ensure the `dmcli_chunks` collection exists with
    /// the correct named-vector schema:
    ///
    /// - `"dense"` — 384-dim Cosine HNSW vectors
    /// - `"sparse"` — sparse vectors (log-TF token weights)
    async fn client(&self) -> Result<qdrant_client::Qdrant> {
        use qdrant_client::{
            Qdrant,
            qdrant::{
                CreateCollectionBuilder, Distance, SparseVectorConfig, SparseVectorParams,
                VectorParamsBuilder, VectorParamsMap, VectorsConfig, vectors_config,
            },
        };

        let url = &self.url;
        let client = Qdrant::from_url(url)
            .build()
            .map_err(|e| Error::Index(format!("Failed to connect to Qdrant at {url}: {e}")))?;

        if !client
            .collection_exists(COLLECTION)
            .await
            .map_err(|e| Error::Index(format!("Qdrant collection_exists failed: {e}")))?
        {
            let dense_params = VectorParamsBuilder::new(MODEL_DIMS, Distance::Cosine).build();
            let vectors_config = VectorsConfig {
                config: Some(vectors_config::Config::ParamsMap(VectorParamsMap {
                    map: [("dense".to_string(), dense_params)].into(),
                })),
            };

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
                .map_err(|e| Error::Index(format!("Failed to create Qdrant collection: {e}")))?;
        }

        Ok(client)
    }
}

#[async_trait]
impl VectorStore for QdrantStore {
    async fn upsert(&self, chunks: Vec<Chunk>) -> Result<()> {
        use qdrant_client::qdrant::{PointStruct, UpsertPointsBuilder, Vector};

        if chunks.is_empty() {
            return Ok(());
        }

        let client = self.client().await?;

        let mut points: Vec<PointStruct> = Vec::with_capacity(chunks.len());
        for chunk in &chunks {
            let mut vectors: HashMap<String, Vector> = HashMap::new();
            vectors.insert("dense".to_string(), chunk.dense.clone().into());
            if !chunk.sparse_indices.is_empty() {
                vectors.insert(
                    "sparse".to_string(),
                    Vector::new_sparse(chunk.sparse_indices.clone(), chunk.sparse_values.clone()),
                );
            }

            let mut payload = qdrant_client::Payload::new();
            payload.insert("content", chunk.text.clone());
            payload.insert("path", chunk.path.to_string_lossy().as_ref());
            payload.insert("section", chunk.section.as_str());
            payload.insert("hash", chunk.hash.as_str());

            // Derive a deterministic u64 point ID from the content hash.
            let id = u64::from_be_bytes(
                Sha256::digest(chunk.hash.as_bytes())[..8]
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

    async fn search(
        &self,
        dense: Vec<f32>,
        sparse_indices: Vec<u32>,
        sparse_values: Vec<f32>,
        max_results: u64,
    ) -> Result<Vec<SearchResult>> {
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

        let client = self.client().await?;

        let dense_prefetch = PrefetchQueryBuilder::default()
            .query(Query::new_nearest(VectorInput::new_dense(dense.clone())))
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
                .query(Query::new_nearest(VectorInput::new_dense(dense)))
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
}
