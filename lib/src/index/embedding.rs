use candle_core::{Device, Tensor};
use candle_transformers::models::bert::{BertModel, DTYPE};
use std::collections::HashMap;
use tokenizers::{EncodeInput, Tokenizer};

use crate::error::Error;
use crate::result::Result;

pub struct EmbeddingBuilder {
    pub device: Device,
    pub tokenizer: Tokenizer,
    pub model: BertModel,
}

impl EmbeddingBuilder {
    /// Extract salient keywords from `prompt` by comparing n-gram candidates
    /// against the global intent vector and returning the top-scoring,
    /// non-redundant phrases above `threshold`.
    pub fn extract_keywords(
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
    pub fn sparse_vector(&self, text: &str) -> Result<(Vec<u32>, Vec<f32>)> {
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

    pub fn encode_dense<'a, E>(&self, input: Vec<E>) -> Result<Vec<Vec<f32>>>
    where
        E: Into<EncodeInput<'a>> + Send,
    {
        let len = input.len();
        if len == 0 {
            return Ok(vec![]);
        }

        let embedding = self.encode(input)?;
        let embedding = embedding.pool_1()?;
        let embedding = embedding.normalize()?;

        let mut result: Vec<Vec<f32>> = Vec::with_capacity(len);
        for i in 0..len {
            result.push(embedding.embedding.get(i).unwrap().try_into().unwrap());
        }
        Ok(result)
    }

    pub fn encode<'a, E>(&self, input: Vec<E>) -> Result<Embedding>
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

pub struct Embedding {
    pub embedding: Tensor,
    pub attention_mask: Tensor,
}

impl Embedding {
    pub fn normalize(self) -> Result<Self> {
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
    pub fn pool_1(self) -> Result<Embedding> {
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
    pub fn pool_2(self) -> Result<Embedding> {
        let (_, s, _) = self.embedding.dims3()?;
        let embedding = (self.embedding.sum(1)? / (s as f64))?.squeeze(0)?;
        Ok(Embedding {
            embedding,
            attention_mask: self.attention_mask,
        })
    }
}
