use crate::errors::Error;
use async_channel::{Receiver, Sender};
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use tokio::io::{AsyncWriteExt, stdout};

use crate::events::AppEvent;

// Include the generated constants from the build script
//include!(concat!(env!("OUT_DIR"), "/model_constants.rs"));

/// A local agent that can be used for inference and text generation.
pub struct LocalAgent {
    client_sender: Sender<AgentAction>,
}

impl LocalAgent {
    /// Creates a new builder for configuring a LocalAgent
    pub fn builder() -> LocalAgentBuilder {
        LocalAgentBuilder::default()
    }
}

impl Drop for LocalAgent {
    fn drop(&mut self) {
        if let Err(e) = self.client_sender.try_send(AgentAction::Poison) {
            panic!("Failed to send poison to internal thread: {e:?}");
        }
    }
}

enum AgentAction {
    Poison,
    Exit,
}

struct AgentLoop {
    app_sender: Sender<AppEvent>,
    client_sender: Sender<AgentAction>,
    client_receiver: Receiver<AgentAction>,
}

impl AgentLoop {
    async fn run_loop(&self) -> Result<(), Error> {
        while let Ok(action) = self.client_receiver.recv().await {
            log::debug!("Got event, updating");

            match action {
                AgentAction::Poison => {
                    self.emit_app_event(AppEvent::AiThinking(
                        "I've been poisoned!".into(),
                        Vec::default(),
                    ));
                    self.add_next_action(AgentAction::Exit);
                }
                AgentAction::Exit => break,
            }
        }

        Ok(())
    }

    fn add_next_action(&self, action: AgentAction) {
        if let Err(e) = self.client_sender.try_send(action) {
            panic!("Failed to send event to internal thread: {e:?}");
        }
    }

    fn emit_app_event(&self, event: AppEvent) {
        if let Err(e) = self.app_sender.try_send(event) {
            panic!("Failed to emit app event: {e:?}");
        }
    }
}

/// Builder for configuring and creating a LocalAgent
#[derive(Default)]
pub struct LocalAgentBuilder {
    model: Option<String>,
    cache_dir: Option<std::path::PathBuf>,
    app_sender: Option<Sender<AppEvent>>,
}

impl LocalAgentBuilder {
    pub fn with_app_sender(self, sender: Sender<AppEvent>) -> Self {
        Self {
            app_sender: Some(sender),
            ..self
        }
    }

    /// Sets a custom model path or identifier
    pub fn with_model(self, model: impl Into<String>) -> Self {
        Self {
            model: Some(model.into()),
            ..self
        }
    }

    /// Sets the cache directory for storing model files.
    ///
    /// When a cache directory is specified, the model files (tokenizer.json,
    /// model.safetensors, and config.json) will be stored in or loaded from this directory.
    /// If the files don't exist, they will be created automatically.
    ///
    /// # Arguments
    ///
    /// * `cache_dir` - Path to the directory where model files should be cached
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::path::Path;
    /// let agent = LocalAgentBuilder::default()
    ///     .with_cache_dir("/tmp/my_cache")
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn with_cache_dir(self, cache_dir: impl Into<std::path::PathBuf>) -> Self {
        Self {
            cache_dir: Some(cache_dir.into()),
            ..self
        }
    }

    /// Builds the LocalAgent with the configured settings
    pub async fn build(self) -> Result<LocalAgent, Error> {
        if let Some(cache_dir) = self.cache_dir {
            log::info!("Using {} for text generation data", cache_dir.display());
        }

        if let Some(model) = self.model {
            log::info!("Loading text generation model {model}");
        }

        log::info!("Downloading model file");
        let api = hf_hub::api::sync::Api::new().expect("Failed to create HF Hub API");
        let model_path = api
            .repo(hf_hub::Repo::with_revision(
                "TheBloke/Llama-2-7B-GGML".to_string(),
                hf_hub::RepoType::Model,
                "main".to_string(),
            ))
            .get("llama-2-7b.ggmlv3.q4_0.bin")
            .expect("Failed to download model file");
        let mut file = std::fs::File::open(&model_path)?;

        log::info!("Initializing model");
        let device = Device::Cpu;
        let gqa = 1;
        let model = candle_core::quantized::ggml_file::Content::read(&mut file, &device)
            .map_err(|e| e.with_path(model_path))
            .expect("failed to read ggml file");
        let mut model =
            candle_transformers::models::quantized_llama::ModelWeights::from_ggml(model, gqa)
                .expect("Can build model");

        log::info!("Downloading tokenizer");
        let tokenizer_path = api
            .model("hf-internal-testing/llama-tokenizer".to_string())
            .get("tokenizer.json")
            .expect("Failed to download tokenizer file");

        log::info!("Initializing tokenizer");
        let tokenizer =
            tokenizers::Tokenizer::from_file(tokenizer_path).expect("Failed to load tokenizer");
        let mut tos = TokenOutputStream::new(tokenizer);

        print!("Finishing query:");
        let prompt_str = "Context: you are a helpful assistant that speaks only english. You will be given requests prefixed with the phrase '{user}' and will provide responses prefixed with '{agent}'. Keep your responses as brief and concise as possible. {user} Describe a dog in five words or less.";
        let temperature = 0.8;
        let sample_len = 1000;
        let seed = 299792458;
        let mut logits_processor = LogitsProcessor::new(seed, Some(temperature), None);

        let eos_token = *tos
            .tokenizer()
            .get_vocab(true)
            .get("</s>")
            .expect("can get eos token");
        log::info!("EOS token: {eos_token}");

        let (prompt_tokens_len, mut tokens) = {
            let tokens = tos
                .tokenizer()
                .encode(prompt_str, true)
                .map_err(|e| panic!("Tokenization failed: {e}"))
                .unwrap();
            let ids = tokens.get_ids();

            for id in ids {
                log::info!(
                    "Selected token {id}: {:?}",
                    tos.tokenizer().decode(&[*id], false)
                );

                if let Some(t) = tos.next_token(*id) {
                    print!("{t}");
                }
            }

            (ids.len(), ids.to_vec())
        };
        let mut index_pos = 0;

        for sample in 0..sample_len {
            log::info!("Processing sample {sample} with {} tokens", tokens.len());

            let tensor = Tensor::new(tokens, &device)
                .expect("can create a new tensor")
                .unsqueeze(0)
                .expect("can unsqueeze");
            log::info!("Created tensor");

            let logits = model
                .forward(&tensor, index_pos)
                .expect("can forward")
                .squeeze(0)
                .expect("can squeeze");
            log::info!("Inferred logits");

            let next_token = logits_processor
                .sample(&logits)
                .expect("can select a token");
            log::info!(
                "Selected token {next_token}: {:?}",
                tos.tokenizer().decode(&[next_token], false)
            );

            if next_token == eos_token {
                log::info!("Reached end of sequence");
                break;
            }

            if let Some(t) = tos.next_token(next_token) {
                print!("{t}");
                stdout().flush().await.expect("can flush stdout");
            }

            tokens = vec![next_token];
            index_pos = prompt_tokens_len + sample;
        }
        if let Some(rest) = tos.decode_rest() {
            println!("{rest}");
            stdout().flush().await.expect("can flush stdout");
        } else {
            println!();
        }

        let (client_sender, client_receiver) = async_channel::unbounded();
        let agent_loop = AgentLoop {
            app_sender: self.app_sender.expect("The app sender channel is required"),
            client_sender: client_sender.clone(),
            client_receiver,
        };

        let _thread = tokio::spawn(async move {
            if let Err(e) = agent_loop.run_loop().await {
                log::error!("Client error: {e}");
            }
        });

        Ok(LocalAgent { client_sender })
    }
}

/// This is a wrapper around a tokenizer to ensure that tokens can be returned to the user in a
/// streaming way rather than having to wait for the full decoding.
pub struct TokenOutputStream {
    tokenizer: tokenizers::Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

#[allow(dead_code)]
impl TokenOutputStream {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn into_inner(self) -> tokenizers::Tokenizer {
        self.tokenizer
    }

    fn decode(&self, tokens: &[u32]) -> String {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => str,
            Err(err) => panic!("cannot decode: {err}"),
        }
    }

    // https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    pub fn next_token(&mut self, token: u32) -> Option<String> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..]);
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Some(text.1.to_string())
        } else {
            None
        }
    }

    pub fn decode_rest(&self) -> Option<String> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)
        };
        let text = self.decode(&self.tokens[self.prev_index..]);
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            Some(text.1.to_string())
        } else {
            None
        }
    }

    pub fn decode_all(&self) -> String {
        self.decode(&self.tokens)
    }

    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token_s).copied()
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_build_with_model() {
        let dir = tempfile::tempdir().unwrap();
        let _ = LocalAgentBuilder::default()
            .with_model("some_model")
            .with_cache_dir(dir.path())
            .build()
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_build_with_temp_dir() {
        let dir = tempfile::tempdir().unwrap();
        let _ = LocalAgentBuilder::default()
            .with_cache_dir(dir.path())
            .build()
            .await
            .unwrap();
    }
}
