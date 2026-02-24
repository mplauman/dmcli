use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{Repo, RepoType, api::sync::Api};
use pulldown_cmark::{Event, HeadingLevel, Parser, Tag, TagEnd};
use sha2::{Digest, Sha256};
use std::fs::read_to_string;
use std::path::Path;
use text_splitter::{ChunkConfig, MarkdownSplitter, TextSplitter};
use tokenizers::{EncodeInput, PaddingParams, Tokenizer};
use walkdir::{DirEntry, WalkDir};

use crate::error::Error;
use crate::result::Result;

const MODEL_NAME: &str = "sentence-transformers/all-MiniLM-L6-v2";
const MODEL_REVISION: &str = "refs/pr/21";
const MODEL_DIMS: usize = 384;

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
    fn extract_keywords<'a>(
        &self,
        prompt: &str,
        max_n: usize,
        top_k: usize,
        threshold: f32,
    ) -> Result<Vec<(String, f32)>> {
        // Pre-process candidates, filtering out stop words that don't mean much.
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
                // Join the window into a single string (e.g., "docker", "docker backup")
                let ngram = window.join(" ");
                candidates.push(ngram);
            }
        }

        if candidates.is_empty() {
            return Ok(vec![]);
        }
        if candidates.len() == 1 {
            candidates.resize(1, String::default());
            return Ok(vec![(candidates.pop().unwrap(), 1.0)]);
        }

        // 4. Get Global Intent (Single Vector)
        //let global_vec = self.encode_new(prompt)?.embedding; // Shape: [HiddenSize]
        let global_vec = self.encode(vec![prompt])?;
        let global_vec = global_vec.pool_2()?;
        let global_vec = global_vec.normalize()?;
        let global_vec = global_vec.embedding;

        let pooled_words = self.encode(candidates.clone())?;
        let pooled_words = pooled_words.pool_2()?;
        let pooled_words = pooled_words.normalize()?;
        let pooled_words = pooled_words.embedding;

        // Matrix multiplication: [N, HiddenSize] * [HiddenSize, 1] -> [N]
        let similarities = pooled_words
            .matmul(&global_vec.unsqueeze(1)?)?
            .flatten_all()?;
        let scores: Vec<f32> = similarities.to_vec1()?;

        // 6. Map back to strings and sort
        let mut results = candidates.into_iter().zip(scores).collect::<Vec<_>>();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.dedup_by(|a, b| a.0 == b.0); // Remove duplicates

        let results = results
            .into_iter()
            .filter(|r| r.1 >= threshold)
            .collect::<Vec<_>>();

        // 1. We assume 'results' is already sorted by score descending
        let mut final_keywords: Vec<(String, f32)> = Vec::new();

        for (phrase, score) in results {
            // Check if this phrase is already "covered" by a better-scoring phrase
            let is_redundant = final_keywords.iter().any(|(existing_phrase, _)| {
                existing_phrase.contains(&phrase) || phrase.contains(existing_phrase)
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

    fn encode<'a, E>(&self, input: Vec<E>) -> Result<Embedding>
    where
        E: Into<EncodeInput<'a>> + Send,
    {
        let len = input.len();
        println!("Encoding {len} inputs");

        if len == 0 {
            return Err(Error::Index("nothing to encode".to_string()));
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

    // Apply avg-pooling by taking the mean embedding value for all
    // tokens (after applying the attention mask from tokenization).
    // This should produce the same numeric result as the
    // `sentence_transformers` Python library.
    fn pool_1(self) -> Result<Embedding> {
        let attention_mask_for_pooling = self.attention_mask.to_dtype(DTYPE)?.unsqueeze(2)?;
        let sum_mask = attention_mask_for_pooling.sum(1)?;
        let embeddings = (self.embedding.broadcast_mul(&attention_mask_for_pooling)?).sum(1)?;
        let embeddings = embeddings.broadcast_div(&sum_mask)?;

        Ok(Embedding {
            embedding: embeddings,
            attention_mask: self.attention_mask,
        })
    }

    // Mean pooling to get a single vector for the text
    fn pool_2(self) -> Result<Embedding> {
        let (_, s, _) = self.embedding.dims3()?;
        let embedding = (self.embedding.sum(1)? / (s as f64))?.squeeze(0)?; // Shape: [N, HiddenSize]

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
                // If we have accumulated text, emit a chunk with the *previous* header stack
                let trimmed = current_text.trim();
                if !trimmed.is_empty() {
                    // Normalize whitespace for the final content
                    let normalized_content =
                        trimmed.split_whitespace().collect::<Vec<&str>>().join(" ");

                    chunks.push(ChunkPayload {
                        content: normalized_content,
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
                // Update header stack
                // If level is 1, stack should have 0 elements before push.
                // If level is 2, stack should have 1 element before push.
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

    // Emit final chunk
    let trimmed = current_text.trim();
    if !trimmed.is_empty() {
        let normalized_content = trimmed.split_whitespace().collect::<Vec<&str>>().join(" ");
        chunks.push(ChunkPayload {
            content: normalized_content,
            path,
            headers: header_stack,
        });
    }

    chunks
}

pub struct DocumentIndex<const CHUNK_SIZE: usize> {
    db: crate::database::Database<MODEL_DIMS>,
    embedding_builder: EmbeddingBuilder,
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
            embedding_builder: EmbeddingBuilder {
                device,
                tokenizer,
                model,
            },
        };

        Ok(result)
    }

    fn split_text<'a>(&self, text: &'a str) -> Result<Vec<&'a str>> {
        let chunk_config = ChunkConfig::new(CHUNK_SIZE)
            .with_sizer(&self.embedding_builder.tokenizer)
            .with_overlap(CHUNK_SIZE / 64)
            .expect("Overlap is a sane value");

        let splitter = TextSplitter::new(chunk_config);
        let chunks: Vec<&str> = splitter.chunks(text).collect();

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

        let embedding = self.embedding_builder.encode(input)?;
        let embedding = embedding.pool_1()?;
        let embedding = embedding.normalize()?;
        let embeddings = embedding.embedding;

        let mut result: Vec<[f32; MODEL_DIMS]> = Vec::new();
        for i in 0..len {
            result.push(
                embeddings
                    .get(i)
                    .unwrap()
                    .to_vec1::<f32>()?
                    .try_into()
                    .unwrap(),
            );
        }

        Ok(result)
    }

    async fn insert(&mut self, texts: Vec<&str>) -> Result<()> {
        let encodings = self.encode(texts.clone())?;

        for (encoding, text) in encodings.into_iter().zip(texts.into_iter()) {
            self.db.insert(&encoding, text).await?;
        }

        Ok(())
    }

    pub async fn search<const MAX_RESULTS: u64>(&self, text: &str) -> Result<Vec<String>> {
        let max_n: usize = 3;
        let top_k: usize = 6;
        let threshold: f32 = 0.45;
        let results = self
            .embedding_builder
            .extract_keywords(text, max_n, top_k, threshold)?;
        println!("Keywords from input {text}: {results:?}");

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
        let chunk_config = ChunkConfig::new(CHUNK_SIZE)
            .with_sizer(&self.embedding_builder.tokenizer)
            .with_overlap(CHUNK_SIZE / 64)
            .expect("Overlap is a sane value");

        let splitter = MarkdownSplitter::new(chunk_config);

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
            let chunks = process_markdown(&contents, entry.as_path());

            let chunks = chunks
                .into_iter()
                .flat_map(|chunk| {
                    splitter
                        .chunks(&chunk.content)
                        .map(|subchunk| subchunk.trim().to_string())
                        .filter(|content| !content.is_empty())
                        .map(|subchunk| {
                            (
                                subchunk.to_string(),
                                chunk.path.to_path_buf(),
                                hex::encode(Sha256::digest(subchunk.as_bytes())),
                                chunk.headers.join("/"),
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let texts = chunks
                .iter()
                .map(|chunk| chunk.0.as_str())
                .collect::<Vec<_>>();
            let embeddings = self.encode(texts)?;

            for (chunk, embedding) in chunks.into_iter().zip(embeddings.into_iter()) {
                println!("Inserting {}#{}", chunk.1.display(), chunk.3);
                self.db
                    .insert_doc_chunk(&embedding, &chunk.0, &chunk.1, &chunk.2, &chunk.3)
                    .await?;
            }
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
        // "Here is a " (Text)
        // "link" (Text from link)
        // " and an " (Text)
        // "image" (Text from image alt)
        // "." (Text)
        // Normalized with spaces:
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
