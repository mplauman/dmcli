use crate::errors::Error;
use llm::chat::{ChatMessage, ChatRole};
use memvdb::{CacheDB, Embedding};
use model2vec_rs::model::StaticModel;
use std::collections::HashMap;

// Include the generated constants from the build script
include!(concat!(env!("OUT_DIR"), "/model_constants.rs"));

pub struct Embeddings {
    cache: CacheDB,
    embedder: StaticModel,
    size: usize,
}

impl Embeddings {
    pub fn builder() -> EmbeddingsBuilder {
        EmbeddingsBuilder::default()
    }

    pub fn embed(
        &mut self,
        messages: impl IntoIterator<Item = impl Into<ChatMessage>>,
    ) -> Result<(), Error> {
        let messages = messages.into_iter().map(|m| m.into()).collect::<Vec<_>>();

        let embeddings = self.embedder.encode(
            &messages
                .iter()
                .map(|m| {
                    let role = match m.role {
                        ChatRole::User => "User",
                        ChatRole::Assistant => "Assistant",
                    };
                    format!("{}: {}", role, m.content)
                })
                .collect::<Vec<String>>(),
        );

        let embeddings = messages
            .iter()
            .zip(embeddings)
            .map(|(msg, vector)| {
                let id = HashMap::from([("index".to_string(), self.size.to_string())]);
                self.size = self.size.saturating_add(1);

                let metadata = Some(HashMap::from([(
                    "content".to_string(),
                    msg.content.clone(),
                )]));

                Embedding {
                    id,
                    vector,
                    metadata,
                }
            })
            .collect::<Vec<Embedding>>();

        self.cache.update_collection("messages", embeddings)?;

        Ok(())
    }
}

#[derive(Default)]
pub struct EmbeddingsBuilder {
    embedding_model: Option<String>,
}

impl EmbeddingsBuilder {
    pub fn with_embedding_model(self, embedding_model: String) -> Self {
        Self {
            embedding_model: Some(embedding_model),
        }
    }

    pub fn build(self) -> Result<Embeddings, Error> {
        let embedding_model_or_path = match self.embedding_model {
            Some(embedding_model) => embedding_model,
            None => {
                let folder = dirs::cache_dir().expect("cache dir exists").join("dmcli");

                if !folder.exists() {
                    std::fs::create_dir_all(&folder)?;
                    std::fs::write(folder.join("tokenizer.json"), TOKENIZER_BYTES)?;
                    std::fs::write(folder.join("model.safetensors"), MODEL_BYTES)?;
                    std::fs::write(folder.join("config.json"), CONFIG_BYTES)?;
                }

                folder.to_string_lossy().into_owned()
            }
        };

        log::info!("Loading Model2Vec model: {embedding_model_or_path}");
        let embedder = StaticModel::from_pretrained(
            embedding_model_or_path,
            None, // No HuggingFace token needed for public models
            None, // Use default normalization from model config
            None, // No subfolder
        )
        .map_err(|e| Error::Embedding(format!("{e}")))?;

        let mut cache = CacheDB::new();
        cache.create_collection(
            "messages".into(),
            embedder.encode_single("test").len(),
            memvdb::Distance::Cosine,
        )?;

        Ok(Embeddings {
            embedder,
            cache,
            size: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm::chat::{ChatMessage, ChatRole, MessageType};
    use tempfile::TempDir;

    fn create_test_embeddings() -> Result<Embeddings, Error> {
        // Create a temporary directory for testing
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model_path = temp_dir.path().to_string_lossy().to_string();

        // Write test model files
        std::fs::write(temp_dir.path().join("tokenizer.json"), TOKENIZER_BYTES)?;
        std::fs::write(temp_dir.path().join("model.safetensors"), MODEL_BYTES)?;
        std::fs::write(temp_dir.path().join("config.json"), CONFIG_BYTES)?;

        Embeddings::builder()
            .with_embedding_model(model_path)
            .build()
    }

    #[test]
    fn test_embeddings_builder_default() {
        let builder = Embeddings::builder();
        assert!(builder.embedding_model.is_none());
    }

    #[test]
    fn test_embeddings_builder_with_model() {
        let model_path = "/test/path".to_string();
        let builder = Embeddings::builder().with_embedding_model(model_path.clone());
        assert_eq!(builder.embedding_model, Some(model_path));
    }

    #[test]
    fn test_embeddings_build_with_custom_model() {
        // This test requires actual model files, so we'll use a mock approach
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model_path = temp_dir.path().to_string_lossy().to_string();

        // Write test model files
        std::fs::write(temp_dir.path().join("tokenizer.json"), TOKENIZER_BYTES).unwrap();
        std::fs::write(temp_dir.path().join("model.safetensors"), MODEL_BYTES).unwrap();
        std::fs::write(temp_dir.path().join("config.json"), CONFIG_BYTES).unwrap();

        let result = Embeddings::builder()
            .with_embedding_model(model_path)
            .build();

        assert!(result.is_ok());
        let embeddings = result.unwrap();
        assert_eq!(embeddings.size, 0);
    }

    #[test]
    fn test_embeddings_build_creates_cache_dir() {
        // Mock the cache directory creation by using a temporary directory
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let cache_path = temp_dir.path().join("test_cache");

        // Manually create the cache directory and files for testing
        std::fs::create_dir_all(&cache_path).unwrap();
        std::fs::write(cache_path.join("tokenizer.json"), TOKENIZER_BYTES).unwrap();
        std::fs::write(cache_path.join("model.safetensors"), MODEL_BYTES).unwrap();
        std::fs::write(cache_path.join("config.json"), CONFIG_BYTES).unwrap();

        let result = Embeddings::builder()
            .with_embedding_model(cache_path.to_string_lossy().to_string())
            .build();

        assert!(result.is_ok());
    }

    #[test]
    fn test_embed_single_message() {
        let mut embeddings = create_test_embeddings().expect("Failed to create embeddings");

        let message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "Hello, world!".to_string(),
        };

        let result = embeddings.embed(vec![message]);
        assert!(result.is_ok());
        assert_eq!(embeddings.size, 1);
    }

    #[test]
    fn test_embed_multiple_messages() {
        let mut embeddings = create_test_embeddings().expect("Failed to create embeddings");

        let messages = vec![
            ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: "Hello, world!".to_string(),
            },
            ChatMessage {
                role: ChatRole::Assistant,
                message_type: MessageType::Text,
                content: "Hi there! How can I help you?".to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: "What's the weather like?".to_string(),
            },
        ];

        let result = embeddings.embed(messages);
        assert!(result.is_ok());
        assert_eq!(embeddings.size, 3);
    }

    #[test]
    fn test_embed_incremental_size() {
        let mut embeddings = create_test_embeddings().expect("Failed to create embeddings");

        // First embedding
        let message1 = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "First message".to_string(),
        };
        embeddings
            .embed(vec![message1])
            .expect("First embed failed");
        assert_eq!(embeddings.size, 1);

        // Second embedding
        let message2 = ChatMessage {
            role: ChatRole::Assistant,
            message_type: MessageType::Text,
            content: "Second message".to_string(),
        };
        embeddings
            .embed(vec![message2])
            .expect("Second embed failed");
        assert_eq!(embeddings.size, 2);

        // Third embedding with multiple messages
        let messages = vec![
            ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: "Third message".to_string(),
            },
            ChatMessage {
                role: ChatRole::Assistant,
                message_type: MessageType::Text,
                content: "Fourth message".to_string(),
            },
        ];
        embeddings.embed(messages).expect("Third embed failed");
        assert_eq!(embeddings.size, 4);
    }

    #[test]
    fn test_embed_different_roles() {
        let mut embeddings = create_test_embeddings().expect("Failed to create embeddings");

        let messages = vec![
            ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: "User message".to_string(),
            },
            ChatMessage {
                role: ChatRole::Assistant,
                message_type: MessageType::Text,
                content: "Assistant message".to_string(),
            },
        ];

        let result = embeddings.embed(messages);
        assert!(result.is_ok());
        assert_eq!(embeddings.size, 2);
    }

    #[test]
    fn test_embed_empty_messages() {
        let mut embeddings = create_test_embeddings().expect("Failed to create embeddings");

        let messages: Vec<ChatMessage> = vec![];
        let result = embeddings.embed(messages);
        assert!(result.is_ok());
        assert_eq!(embeddings.size, 0);
    }

    #[test]
    fn test_embed_with_empty_content() {
        let mut embeddings = create_test_embeddings().expect("Failed to create embeddings");

        let message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "".to_string(),
        };

        let result = embeddings.embed(vec![message]);
        assert!(result.is_ok());
        assert_eq!(embeddings.size, 1);
    }

    #[test]
    fn test_embed_with_special_characters() {
        let mut embeddings = create_test_embeddings().expect("Failed to create embeddings");

        let message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "Hello! @#$%^&*()_+ 你好 🚀".to_string(),
        };

        let result = embeddings.embed(vec![message]);
        assert!(result.is_ok());
        assert_eq!(embeddings.size, 1);
    }

    #[test]
    fn test_embed_with_long_content() {
        let mut embeddings = create_test_embeddings().expect("Failed to create embeddings");

        let long_content = "a".repeat(10000);
        let message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: long_content,
        };

        let result = embeddings.embed(vec![message]);
        assert!(result.is_ok());
        assert_eq!(embeddings.size, 1);
    }

    #[test]
    fn test_size_saturation() {
        let mut embeddings = create_test_embeddings().expect("Failed to create embeddings");

        // Set size close to max value to test saturation
        embeddings.size = usize::MAX - 1;

        let message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "Test saturation".to_string(),
        };

        let result = embeddings.embed(vec![message]);
        assert!(result.is_ok());
        assert_eq!(embeddings.size, usize::MAX);

        // Add another message to test that it stays at MAX
        let message2 = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "Test saturation 2".to_string(),
        };

        let result2 = embeddings.embed(vec![message2]);
        assert!(result2.is_ok());
        assert_eq!(embeddings.size, usize::MAX);
    }
}
