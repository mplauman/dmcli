use crate::errors::Error;

// Include the generated constants from the build script
//include!(concat!(env!("OUT_DIR"), "/model_constants.rs"));

/// A local agent that can be used for inference and text generation.
pub struct LocalAgent {}

impl LocalAgent {
    /// Creates a new builder for configuring a LocalAgent
    pub fn builder() -> LocalAgentBuilder {
        LocalAgentBuilder::default()
    }
}

/// Builder for configuring and creating a LocalAgent
#[derive(Default)]
pub struct LocalAgentBuilder {
    model: Option<String>,
    cache_dir: Option<std::path::PathBuf>,
}

impl LocalAgentBuilder {
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

        log::info!("Initialized local text generation agent");

        Ok(LocalAgent {})
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
