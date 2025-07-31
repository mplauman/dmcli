use crate::conversation::{Conversation, Message};
use crate::errors::Error;
use crate::events::AppEvent;
use async_channel::{Receiver, Sender};
use candle_core::quantized::gguf_file::Content;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_phi3::ModelWeights;
use tokenizers::Tokenizer;

// Include the generated constants from the build script
//include!(concat!(env!("OUT_DIR"), "/model_constants.rs"));

static SYSTEM_PROMPT: &str = "
You are a dungeon master's helpful assistant. You're role is to help them search through their notes to
provide them with information and backstory to help them prepare for games. You will also help them
create NPCs, monsters, scenes, situations, and descriptions of fantastic items based on the style of
the dungeon master's session notes.

## Communication

1. Keep your responses short. Avoid unnecessary details or tangents.
2. Don't apologize if you're unable to do something. Do your best, and explain why if you are unable to proceed.
3. As much as possible, base your answers on the DM's notes. Read multiple files to gain context.
4. Bias towards not asking the user for help if you can find the answer yourself.
5. When providing paths to tools, the path should always begin with a path that starts with a project root directory listed above.
6. Before you read or edit a file, you must first find the full path. DO NOT ever guess a file path!

## Tool Usage

1. When multiple tools can be used independently to solve different parts of a request, use them in parallel rather than sequentially.
2. For example, if you need to search for both a character and a location, make both tool calls at once rather than one after the other.
3. For dependent tools (where one tool's output is needed for another), use them sequentially as required.
4. Always try to minimize the number of back-and-forth interactions by batching independent tool calls.

## Searching Notes

When searching through notes, first get a list of all the potentially interesting files by listing them, grepping through
them, or both. When you have a list of candidates, read their contents and use that to inform your answer. Most of the time
the information you will need is spread through multiple files.

## Monster Generation

Immediately generate the monster stat block AND ONLY the monster stat block. Do not generate any remarks before or after the stat block. Wrap the stat block in a code block and format it as markdown.

Use this stat block format for monsters:
```
| STR | DEX | CON | INT | WIS | CHA |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 19 (+4) | 12 (+1) | 18 (+4) | 12 (+1) | 16 (+3) | 14 (+2) |
```
";

/// A local agent that can be used for inference and text generation.
pub struct LocalAgent {
    client_sender: Sender<AgentAction>,
}

impl LocalAgent {
    /// Creates a new builder for configuring a LocalAgent
    pub fn builder() -> LocalAgentBuilder {
        LocalAgentBuilder::default()
    }

    pub fn push(&mut self, conversation: &Conversation) -> Result<(), Error> {
        if let Some(Message::User { content, .. }) = conversation.into_iter().next() {
            self.client_sender
                .try_send(AgentAction::Chat(content.clone()))
                .expect("The client sender channel is still open");
        };

        Ok(())
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
    Chat(String),
    Initialize,
    Poison,
}

struct AgentLoop {
    model: Option<ModelWeights>,
    tos: Option<Tokenizer>,
    processor: LogitsProcessor,
    max_sample_len: usize,
    eos_token: u32,
    app_sender: Sender<AppEvent>,
    client_sender: Sender<AgentAction>,
    client_receiver: Receiver<AgentAction>,
}

impl AgentLoop {
    async fn run_loop(&mut self) -> Result<(), Error> {
        while let Ok(action) = self.client_receiver.recv().await {
            log::debug!("Got event, updating");

            match action {
                AgentAction::Chat(text) => self.chat(&text).await?,
                AgentAction::Initialize => self.initialize().await?,
                AgentAction::Poison => break,
            }
        }

        Ok(())
    }

    async fn chat(&mut self, text: &str) -> Result<(), Error> {
        let prompt_str =
            format!("<|system|>\n{SYSTEM_PROMPT}\n<|end|><|user|>\n{text}<|end|>\n<|assistant|>");

        let tos = self.tos.as_ref().expect("tos has been set up");

        log::info!("Cloning model...");
        let mut model = self.model.as_ref().expect("model has been set up").clone();
        log::info!("Cloning done");

        let mut tokens = tos
            .encode(prompt_str, true)
            .map_err(|e| panic!("Tokenization failed: {e}"))
            .unwrap()
            .get_ids()
            .to_vec();

        let mut response = Vec::default();

        let mut index_pos = 0;
        for sample in 0..self.max_sample_len {
            log::info!("Processing sample {sample} with {} tokens", tokens.len());

            let tensor = Tensor::new(tokens.as_slice(), &Device::Cpu)
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

            let next_token = self.processor.sample(&logits).expect("can select a token");
            log::info!(
                "Selected token {next_token}: {:?}",
                tos.decode(&[next_token], false)
            );

            response.push(next_token);

            if next_token == self.eos_token {
                log::info!("Reached end of sequence");
                break;
            }

            index_pos += tokens.len();
            tokens = vec![next_token];
        }

        let decoded = tos.decode(&response, true).expect("can decode tokens");

        self.emit_app_event(AppEvent::AiResponse(decoded));

        Ok(())
    }

    async fn initialize(&mut self) -> Result<(), Error> {
        log::info!("Downloading model file");
        let api = hf_hub::api::sync::Api::new().expect("Failed to create HF Hub API");
        let model_path = api
            .repo(hf_hub::Repo::with_revision(
                "microsoft/Phi-3-mini-4k-instruct-gguf".to_string(),
                hf_hub::RepoType::Model,
                "main".to_string(),
            ))
            .get("Phi-3-mini-4k-instruct-q4.gguf")
            .expect("Failed to download model file");
        let mut file = std::fs::File::open(&model_path)?;

        log::info!("Initializing model");
        let model = Content::read(&mut file)
            .map_err(|e| e.with_path(model_path))
            .expect("failed to read gguf file");
        self.model = Some(
            ModelWeights::from_gguf(false, model, &mut file, &Device::Cpu)
                .expect("Can build model"),
        );

        log::info!("Downloading tokenizer");
        let tokenizer_path = api
            .model("microsoft/Phi-3-mini-4k-instruct".to_string())
            .get("tokenizer.json")
            .expect("Failed to download tokenizer file");

        log::info!("Initializing tokenizer");
        let tokenizer =
            tokenizers::Tokenizer::from_file(tokenizer_path).expect("Failed to load tokenizer");

        self.eos_token = *tokenizer
            .get_vocab(true)
            .get("<|end|>")
            .expect("can get eos token");
        log::info!("EOS token {}", self.eos_token);

        self.tos = Some(tokenizer);

        self.emit_app_event(AppEvent::System("Finished initializing local agent".into()));

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
pub struct LocalAgentBuilder {
    app_sender: Option<Sender<AppEvent>>,
    temperature: f64,
    max_sample_len: usize,
    top_p: f64,
    seed: u64,
}

impl Default for LocalAgentBuilder {
    fn default() -> Self {
        Self {
            app_sender: None,
            temperature: 0.8,
            max_sample_len: 1024,
            top_p: 0.7,
            seed: 299792458,
        }
    }
}

impl LocalAgentBuilder {
    pub fn with_app_sender(self, sender: Sender<AppEvent>) -> Self {
        Self {
            app_sender: Some(sender),
            ..self
        }
    }

    pub fn with_temperature(self, temperature: f64) -> Self {
        Self {
            temperature,
            ..self
        }
    }

    pub fn with_max_sample_len(self, max_sample_len: usize) -> Self {
        Self {
            max_sample_len,
            ..self
        }
    }

    pub fn with_top_p(self, top_p: f64) -> Self {
        Self { top_p, ..self }
    }

    pub fn with_seed(self, seed: u64) -> Self {
        Self { seed, ..self }
    }

    /// Builds the LocalAgent with the configured settings
    pub async fn build(self) -> Result<LocalAgent, Error> {
        let (client_sender, client_receiver) = async_channel::unbounded();
        let mut agent_loop = AgentLoop {
            model: None,
            tos: None,
            eos_token: 0,
            max_sample_len: self.max_sample_len,
            processor: LogitsProcessor::new(self.seed, Some(self.temperature), Some(self.top_p)),
            app_sender: self.app_sender.expect("The app sender channel is required"),
            client_sender: client_sender.clone(),
            client_receiver,
        };
        agent_loop.add_next_action(AgentAction::Initialize);

        let _thread = tokio::spawn(async move {
            if let Err(e) = agent_loop.run_loop().await {
                log::error!("Client error: {e}");
            }
        });

        Ok(LocalAgent { client_sender })
    }
}
