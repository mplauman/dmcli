use crate::errors::Error;
use model2vec_rs::model::StaticModel;
use std::time::{Duration, SystemTime};

use serde_json::Value;

// Include the generated constants from the build script
include!(concat!(env!("OUT_DIR"), "/model_constants.rs"));

/// A unique identifier for a message in a conversation
///
/// # Fields
///
/// * `conversation` - The system time when the conversation started
/// * `timestamp` - The duration since the conversation start when this message was created
#[derive(Hash, PartialEq, Eq, Clone)]
pub struct Id {
    conversation: SystemTime,
    timestamp: Duration,
}

impl Id {
    fn new(conversation: SystemTime) -> Self {
        Self {
            conversation,
            timestamp: SystemTime::now()
                .duration_since(conversation)
                .expect("timestamp math works"),
        }
    }
}

impl std::fmt::Display for Id {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let timestamp = self
            .conversation
            .checked_add(self.timestamp)
            .expect("time is reasonable");

        write!(f, "{timestamp:?}")
    }
}

/// A message in a conversation
///
/// # Variants
///
/// * `User { id, content }` - A message from the user
///   - `id`: Unique identifier for this message
///   - `content`: The text content of the user's message
/// * `Assistant { id, content }` - A response from the AI assistant
///   - `id`: Unique identifier for this message
///   - `content`: The text content of the assistant's response
/// * `Thinking { id, content }` - A message indicating the assistant is processing
///   - `id`: Unique identifier for this message
///   - `content`: Description of what the assistant is thinking about
/// * `System { id, content }` - A message from the system
///   - `id`: Unique identifier for this message
///   - `content`: System message content or command output
/// * `Error { id, content }` - An error message
///   - `id`: Unique identifier for this message
///   - `content`: Description of the error that occurred
pub enum Message {
    User {
        id: Id,
        content: String,
        encoding: Vec<f32>,
    },
    Assistant {
        id: Id,
        content: String,
        encoding: Vec<f32>,
    },
    Thinking {
        id: Id,
        content: String,
        tools: Vec<(String, String, Value)>,
    },
    ThinkingDone {
        id: Id,
        tools: Vec<(String, String, String, Vec<f32>)>,
    },
    System {
        id: Id,
        content: String,
    },
    Error {
        id: Id,
        content: String,
    },
}

impl std::fmt::Display for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Message::User { id, content, .. } => write!(f, "user {id}: {content}"),
            Message::Assistant { id, content, .. } => write!(f, "assistant {id}: {content}"),
            Message::Thinking { id, content, .. } => write!(f, "thinking {id}: {content}"),
            Message::ThinkingDone { id, .. } => write!(f, "thinking done {id}"),
            Message::System { id, content } => write!(f, "system {id}: {content}"),
            Message::Error { id, content } => write!(f, "error {id}: {content}"),
        }
    }
}

/// Represents a sequence of messages in an ongoing conversation
///
/// A conversation is a collection of messages with different roles (user, assistant, system etc.)
/// that are created in sequential order. Each message is timestamped relative to when the
/// conversation started.
///
/// # Example
///
/// ```
/// let mut conversation = Conversation::default();
///
/// // Add a user message
/// conversation.user("What is the capital of France?");
///
/// // Add an assistant response
/// conversation.assistant("The capital of France is Paris.");
///
/// // Add a system message
/// conversation.system("`[4] = **4**");
/// ```
pub struct Conversation {
    id: SystemTime,
    messages: Vec<Message>,
    embedder: StaticModel,
}

impl Conversation {
    pub fn builder() -> ConversationBuilder {
        ConversationBuilder::default()
    }

    pub fn user(&mut self, content: impl Into<String>) {
        let content = content.into();
        let encoding = self.encode(&content);

        self.messages.push(Message::User {
            id: Id::new(self.id),
            content,
            encoding,
        });
    }

    pub fn assistant(&mut self, content: impl Into<String>) {
        let content = content.into();
        let encoding = self.encode(&content);

        self.messages.push(Message::Assistant {
            id: Id::new(self.id),
            content,
            encoding,
        });
    }

    pub fn system(&mut self, content: impl Into<String>) {
        self.messages.push(Message::System {
            id: Id::new(self.id),
            content: content.into(),
        });
    }

    pub fn thinking(
        &mut self,
        content: impl Into<String>,
        tools: impl IntoIterator<Item = (String, String, Value)>,
    ) {
        self.messages.push(Message::Thinking {
            id: Id::new(self.id),
            content: content.into(),
            tools: tools.into_iter().collect(),
        });
    }

    pub fn thinking_done(&mut self, tools: impl IntoIterator<Item = (String, String, String)>) {
        self.messages.push(Message::ThinkingDone {
            id: Id::new(self.id),
            tools: tools
                .into_iter()
                .map(|t| {
                    let content = t.2;
                    let encoding = self.encode(&content);

                    (t.0, t.1, content, encoding)
                })
                .collect(),
        });
    }

    pub fn error(&mut self, content: impl Into<String>) {
        self.messages.push(Message::Error {
            id: Id::new(self.id),
            content: content.into(),
        });
    }

    fn encode(&self, content: &str) -> Vec<f32> {
        self.embedder.encode_single(content)
    }
}

impl<'a> IntoIterator for &'a Conversation {
    type Item = &'a Message;
    type IntoIter = std::iter::Rev<std::slice::Iter<'a, Message>>;

    fn into_iter(self) -> Self::IntoIter {
        self.messages.iter().rev()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl Message {
        fn content(&self) -> &str {
            match self {
                Message::User { content, .. } => content,
                Message::Assistant { content, .. } => content,
                Message::System { content, .. } => content,
                Message::Thinking { content, .. } => content,
                Message::ThinkingDone { .. } => todo!("implement this"),
                Message::Error { content, .. } => content,
            }
        }
    }

    #[test]
    fn test_into_iterator_implementation() {
        let mut conversation = ConversationBuilder::default().build().unwrap();
        conversation.user("Hello");
        conversation.assistant("Hi there!");
        conversation.system("Connection established");

        let messages: Vec<&Message> = (&conversation).into_iter().collect();

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].content(), "Connection established");
        assert_eq!(messages[1].content(), "Hi there!");
        assert_eq!(messages[2].content(), "Hello");

        // Test that we can iterate multiple times
        let count = (&conversation).into_iter().count();
        assert_eq!(count, 3);
    }
}

#[derive(Default)]
pub struct ConversationBuilder {
    embedding_model: Option<String>,
}

impl ConversationBuilder {
    pub fn with_embedding_model(self, model: String) -> Self {
        let embedding_model = Some(model);
        Self { embedding_model }
    }

    pub fn build(self) -> Result<Conversation, Error> {
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

        Ok(Conversation {
            id: SystemTime::now(),
            messages: Vec::new(),
            embedder,
        })
    }
}
