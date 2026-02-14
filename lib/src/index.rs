use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{Repo, RepoType, api::sync::Api};
use std::fs::read_to_string;
use std::path::Path;
use text_splitter::{ChunkConfig, MarkdownSplitter, TextSplitter};
use tokenizers::{EncodeInput, PaddingParams, Tokenizer};
use walkdir::{DirEntry, WalkDir};

use crate::error::Error;
use crate::result::Result;

const MODEL_NAME: &'static str = "sentence-transformers/all-MiniLM-L6-v2";
const MODEL_REVISION: &'static str = "refs/pr/21";
const MODEL_DIMS: usize = 384;

pub struct DocumentIndex<const CHUNK_SIZE: usize> {
    db: crate::database::Database<MODEL_DIMS>,
    device: Device,
    tokenizer: Tokenizer,
    model: BertModel,
}

impl<const CHUNK_SIZE: usize> DocumentIndex<CHUNK_SIZE> {
    pub async fn new() -> Result<Self> {
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
            println!("Updating tokenizer padding strategy to BatchLongest");
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            println!("Creating tokenization padding strategy with BatchLongest");
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };

        let model = BertModel::load(vb, &config)?;

        let result = Self {
            db: crate::database::Database::<MODEL_DIMS>::new().await?,
            device,
            tokenizer,
            model,
        };

        Ok(result)
    }

    fn split_text<'a>(&self, text: &'a str) -> Result<Vec<&'a str>> {
        let chunk_config = ChunkConfig::new(CHUNK_SIZE)
            .with_sizer(&self.tokenizer)
            .with_overlap(CHUNK_SIZE / 64)
            .expect("Overlap is a sane value");

        let splitter = TextSplitter::new(chunk_config);
        let chunks: Vec<&str> = splitter.chunks(text).collect();

        Ok(chunks)
    }

    fn split_markdwon<'a>(&self, markdown: &'a str) -> Result<Vec<&'a str>> {
        let chunk_config = ChunkConfig::new(CHUNK_SIZE)
            .with_sizer(&self.tokenizer)
            .with_overlap(CHUNK_SIZE / 64)
            .expect("Overlap is a sane value");

        let splitter = MarkdownSplitter::new(chunk_config);
        let chunks: Vec<&str> = splitter.chunks(markdown).collect();

        Ok(chunks)
    }

    fn encode<'a, E>(&self, input: Vec<E>) -> Result<Vec<[f32; MODEL_DIMS]>>
    where
        E: Into<EncodeInput<'a>> + Send,
    {
        let len = input.len();
        println!("Encoding {len} inputs");

        if len == 0 {
            return Ok(vec![]);
        }

        let tokens = self
            .tokenizer
            .encode_batch(input, true)
            .map_err(|e| Error::Index(format!("Failed to encode batch: {e:?}")))?;

        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Ok(Tensor::new(tokens.as_slice(), &self.device)?)
            })
            .collect::<Result<Vec<_>>>()?;

        let attention_mask = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_attention_mask().to_vec();
                Ok(Tensor::new(tokens.as_slice(), &self.device)?)
            })
            .collect::<Result<Vec<_>>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;
        let token_type_ids = token_ids.zeros_like()?;

        let embeddings = self
            .model
            .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

        let embeddings = {
            // Apply avg-pooling by taking the mean embedding value for all
            // tokens (after applying the attention mask from tokenization).
            // This should produce the same numeric result as the
            // `sentence_transformers` Python library.
            let attention_mask_for_pooling = attention_mask.to_dtype(DTYPE)?.unsqueeze(2)?;
            let sum_mask = attention_mask_for_pooling.sum(1)?;
            let embeddings = (embeddings.broadcast_mul(&attention_mask_for_pooling)?).sum(1)?;
            embeddings.broadcast_div(&sum_mask)?
        };

        let embeddings = DocumentIndex::<CHUNK_SIZE>::normalize_l2(&embeddings)?;

        let mut result: Vec<[f32; MODEL_DIMS]> = Vec::new();
        for i in 0..len {
            result.push(
                embeddings
                    .get(i)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
                    .try_into()
                    .unwrap(),
            );
        }

        Ok(result)
    }

    fn normalize_l2(v: &Tensor) -> Result<Tensor> {
        Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
    }

    async fn insert(&mut self, texts: Vec<&str>) -> Result<()> {
        let encodings = self.encode(texts.clone())?;

        for (encoding, text) in encodings.into_iter().zip(texts.into_iter()) {
            self.db.insert(&encoding, text).await?;
        }

        Ok(())
    }

    pub async fn search<const MAX_RESULTS: u64>(&self, text: &str) -> Result<Vec<String>> {
        let chunks = self.split_text(text)?;

        let encodings = self.encode(chunks)?;

        let mut results = Vec::new();
        for encoding in encodings {
            let found = self.db.find_similar(&encoding, MAX_RESULTS).await?;
            results.extend(found.into_iter());
        }

        Ok(results)
    }

    pub async fn index_path(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let walker = WalkDir::new(path)
            .into_iter()
            .filter_entry(|e| {
                let hidden = e
                    .file_name()
                    .to_str()
                    .map(|name| name.starts_with("."))
                    .unwrap_or(false);

                !hidden
            })
            .filter_map(|e| e.map(DirEntry::into_path).ok())
            .filter(|e| e.is_file())
            .filter(|e| e.extension().and_then(|e| e.to_str()) == Some("md"));

        for entry in walker {
            let contents = read_to_string(&entry)?;
            let chunks = self.split_markdwon(&contents)?;

            println!("Split {} into {}", entry.display(), chunks.len());

            self.insert(chunks).await?;
        }

        Ok(())
    }

    pub async fn index_str(&mut self, text: &str) -> Result<()> {
        let chunks = self.split_text(text)?;

        self.insert(chunks).await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn simple_search() {
        let index = DocumentIndex::<5>::new().await.unwrap();

        let results = index.search::<10>("Hello, world!").await.unwrap();

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn encode_sentences() {
        let mut index = DocumentIndex::<30>::new().await.unwrap();

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
}
