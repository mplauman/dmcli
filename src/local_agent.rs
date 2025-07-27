use crate::errors::Error;
use async_channel::{Receiver, Sender};

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
