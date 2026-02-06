use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{Repo, RepoType, api::sync::Api};
use std::fs::read_to_string;
use text_splitter::{ChunkConfig, MarkdownSplitter, TextSplitter};
use tokenizers::{EncodeInput, Tokenizer};
use walkdir::{DirEntry, WalkDir};

use crate::error::Error;
use crate::result::Result;

pub struct DocumentIndex<const CHUNK_SIZE: usize> {
    db: crate::database::Database<CHUNK_SIZE>,
    tokenizer: Tokenizer,
    _model: BertModel,
    _tok: Tokenizer,
}

impl<const CHUNK_SIZE: usize> DocumentIndex<CHUNK_SIZE> {
    pub async fn new() -> Result<Self> {
        let device = Device::Cpu;
        let model_id = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        let revision = "refs/pr/21".to_string();

        let repo = Repo::with_revision(model_id, RepoType::Model, revision);
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
        let tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| Error::Index(format!("Failed to initialize tokenizer: {e:?}")))?;

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };

        let model = BertModel::load(vb, &config)?;

        let result = Self {
            db: crate::database::Database::<CHUNK_SIZE>::new().await?,
            tokenizer: Tokenizer::from_pretrained("bert-base-uncased", None)
                .map_err(|e| Error::Index(format!("Failed to initialize tokenizer: {e:?}")))?,
            _model: model,
            _tok: tokenizer,
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

    fn encode<'a, E>(&self, input: Vec<E>) -> Result<Vec<[f32; CHUNK_SIZE]>>
    where
        E: Into<EncodeInput<'a>> + Send,
    {
        let encodings: Vec<[f32; CHUNK_SIZE]> = self
            .tokenizer
            .encode_batch_fast(input, false)
            .map_err(|e| Error::Index(format!("Failed to encode batch: {e:?}")))?
            .into_iter()
            .map(|e| {
                let ids = e.get_ids();
                let sum: u32 = ids.iter().sum();
                let sum = sum as f32;

                let mut normalized = ids.iter().map(|id| *id as f32 / sum).collect::<Vec<_>>();

                normalized.resize(CHUNK_SIZE, 0.0);

                normalized
                    .try_into()
                    .expect("Chunking should never return too many tokens")
            })
            .collect();

        Ok(encodings)
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
        println!(" => {} chunks", chunks.len());

        let encodings = self.encode(chunks)?;

        let mut results = Vec::new();
        for encoding in encodings {
            let found = self.db.find_similar(&encoding, MAX_RESULTS).await?;
            results.extend(found.into_iter());
        }

        Ok(results)
    }

    pub async fn index(&mut self, path: &str) -> Result<IndexStatus> {
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
            println!("Processing {entry:?}");

            let contents = read_to_string(&entry)?;
            let chunks = self.split_markdwon(&contents)?;
            println!(" => {} chunks", chunks.len());

            self.insert(chunks).await?;
        }

        Ok(IndexStatus::Complete(path.to_string()))
    }

    pub async fn index_str(&mut self, text: &str) -> Result<IndexStatus> {
        let chunks = self.split_text(text)?;
        println!(" => {} chunks", chunks.len());

        self.insert(chunks).await?;

        Ok(IndexStatus::Complete(text.to_string()))
    }
}

pub enum IndexStatus {
    Complete(String),
    InProgress(String),
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
}
