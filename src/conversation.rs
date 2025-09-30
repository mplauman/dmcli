use crate::embeddings::EmbeddingGenerator;
use crate::errors::Error;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use serde_json::Value;

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
    embedder: Arc<dyn EmbeddingGenerator>,
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
        self.embedder.encode(content)
    }

    pub fn related(&self, skip: usize, content: &str, max: usize) -> Vec<Message> {
        let target = self.encode(content);

        let mut heap = std::collections::BinaryHeap::new();

        for message in self.into_iter().skip(skip) {
            let distances = match message {
                Message::User { encoding, .. } => vec![self.embedder.distance(&target, encoding)],
                Message::Assistant { encoding, .. } => {
                    vec![self.embedder.distance(&target, encoding)]
                }
                Message::Thinking { .. } => continue,
                Message::ThinkingDone { tools, .. } => tools
                    .iter()
                    .map(|t| self.embedder.distance(&target, &t.3))
                    .collect(),
                Message::System { .. } => continue,
                Message::Error { .. } => continue,
            };

            for (i, distance) in distances.into_iter().enumerate() {
                let message = match message {
                    Message::User {
                        id,
                        content,
                        encoding,
                    } => Message::User {
                        id: id.clone(),
                        content: content.clone(),
                        encoding: encoding.clone(),
                    },
                    Message::Assistant {
                        id,
                        content,
                        encoding,
                    } => Message::Assistant {
                        id: id.clone(),
                        content: content.clone(),
                        encoding: encoding.clone(),
                    },
                    Message::Thinking { .. } => panic!("should not be ranked"),
                    Message::ThinkingDone { id, tools } => Message::ThinkingDone {
                        id: id.clone(),
                        tools: vec![tools[i].clone()],
                    },
                    Message::System { .. } => panic!("should not be ranked"),
                    Message::Error { .. } => panic!("should not be ranked"),
                };

                heap.push(RankedMessage { message, distance });
            }

            while heap.len() > max {
                heap.pop();
            }
        }

        heap.into_iter()
            .map(|r| r.message)
            .collect::<Vec<Message>>()
    }
}

struct RankedMessage {
    message: Message,
    distance: f32,
}

impl Ord for RankedMessage {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.partial_cmp(&other.distance).unwrap()
    }
}

impl PartialOrd for RankedMessage {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for RankedMessage {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for RankedMessage {}

impl<'a> IntoIterator for &'a Conversation {
    type Item = &'a Message;
    type IntoIter = std::iter::Rev<std::slice::Iter<'a, Message>>;

    fn into_iter(self) -> Self::IntoIter {
        self.messages.iter().rev()
    }
}

#[derive(Default)]
pub struct ConversationBuilder {
    embedder: Option<Arc<dyn EmbeddingGenerator>>,
}

impl ConversationBuilder {
    /// Sets the embedding generator instance to use for the conversation
    ///
    /// # Arguments
    ///
    /// * `embedder` - An Arc-wrapped instance of EmbeddingGenerator
    ///
    /// # Example
    ///
    /// ```rust
    /// use dmcli::embeddings::EmbeddingGeneratorBuilder;
    /// use dmcli::conversation::ConversationBuilder;
    /// use std::sync::Arc;
    ///
    /// let embedder = Arc::new(EmbeddingGeneratorBuilder::default().build().unwrap());
    /// let conversation = ConversationBuilder::default()
    ///     .with_embedder(embedder)
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn with_embedder(self, embedder: Arc<dyn EmbeddingGenerator>) -> Self {
        Self {
            embedder: Some(embedder),
        }
    }

    pub fn build(self) -> Result<Conversation, Error> {
        let embedder = self.embedder.ok_or_else(|| {
            Error::Embedding(
                "No embedding generator provided. Use with_embedder() to set one.".to_string(),
            )
        })?;

        Ok(Conversation {
            id: SystemTime::now(),
            messages: Vec::new(),
            embedder,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::EmbeddingGenerator;
    use crate::embeddings::TestEmbedder;
    use std::sync::Arc;

    /// Helper method to create a new conversation for testing
    fn create_test_conversation() -> Conversation {
        let embedder: Arc<dyn EmbeddingGenerator> = Arc::new(TestEmbedder {});
        ConversationBuilder::default()
            .with_embedder(embedder)
            .build()
            .unwrap()
    }

    impl Message {
        fn content(&self) -> String {
            match self {
                Message::User { content, .. } => content.clone(),
                Message::Assistant { content, .. } => content.clone(),
                Message::System { content, .. } => content.clone(),
                Message::Thinking { content, .. } => content.clone(),
                Message::ThinkingDone { tools, .. } => {
                    // For testing purposes, return a summary of tool outputs
                    tools
                        .iter()
                        .map(|(_, _, content, _)| content.as_str())
                        .collect::<Vec<_>>()
                        .join("; ")
                }
                Message::Error { content, .. } => content.clone(),
            }
        }
    }

    #[test]
    fn test_into_iterator_implementation() {
        let mut conversation = create_test_conversation();
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

    #[test]
    fn test_related_empty_conversation() {
        let conversation = create_test_conversation();
        let related = conversation.related(0, "test query", 5);
        assert_eq!(related.len(), 0);
    }

    #[test]
    fn test_related_max_zero() {
        let mut conversation = create_test_conversation();
        conversation.user("Hello world");
        conversation.assistant("Hi there");

        let related = conversation.related(0, "greeting", 0);
        assert_eq!(related.len(), 0);
    }

    #[test]
    fn test_related_respects_max_limit() {
        let mut conversation = create_test_conversation();
        conversation.user("First message");
        conversation.assistant("First response");
        conversation.user("Second message");
        conversation.assistant("Second response");
        conversation.user("Third message");

        let related = conversation.related(0, "message", 3);
        assert!(related.len() <= 3);

        let related = conversation.related(0, "message", 2);
        assert!(related.len() <= 2);
    }

    #[test]
    fn test_related_only_includes_rankable_messages() {
        let mut conversation = create_test_conversation();
        conversation.user("User message");
        conversation.assistant("Assistant message");
        conversation.system("System message");
        conversation.error("Error message");
        conversation.thinking("Thinking content", vec![]);

        let related = conversation.related(0, "message", 10);

        // Should only include User and Assistant messages (2 total)
        // System, Error, and Thinking messages should be excluded
        assert_eq!(related.len(), 2);

        for message in related {
            match message {
                Message::User { .. } | Message::Assistant { .. } => {}
                _ => panic!("Only User and Assistant messages should be returned"),
            }
        }
    }

    #[test]
    fn test_related_includes_thinking_done_tools() {
        let mut conversation = create_test_conversation();
        conversation.user("User message");
        conversation.thinking_done(vec![
            (
                "tool1".to_string(),
                "name1".to_string(),
                "Tool output 1".to_string(),
            ),
            (
                "tool2".to_string(),
                "name2".to_string(),
                "Tool output 2".to_string(),
            ),
        ]);

        let related = conversation.related(0, "tool output", 10);

        // Should include the user message plus potentially both tool outputs
        assert!(!related.is_empty());
        assert!(related.len() <= 3); // 1 user + 2 possible tool outputs

        // Check that we can get ThinkingDone messages
        let has_thinking_done = related
            .iter()
            .any(|m| matches!(m, Message::ThinkingDone { .. }));
        assert!(has_thinking_done);
    }

    #[test]
    fn test_related_message_content_preserved() {
        let mut conversation = create_test_conversation();
        let user_content = "Hello world";
        let assistant_content = "Hi there friend";

        conversation.user(user_content);
        conversation.assistant(assistant_content);

        let related = conversation.related(0, "hello", 10);

        assert_eq!(related.len(), 2);

        // Check that the content is preserved correctly
        let contents: Vec<String> = related.iter().map(|m| m.content()).collect();
        assert!(contents.contains(&user_content.to_string()));
        assert!(contents.contains(&assistant_content.to_string()));
    }

    #[test]
    fn test_related_different_similarities() {
        let mut conversation = create_test_conversation();

        // Add messages with different expected similarities to "cat"
        conversation.user("I love cats and dogs"); // Should be most similar
        conversation.assistant("The weather is nice today"); // Should be least similar
        conversation.user("My cat is very fluffy"); // Should be very similar

        let related = conversation.related(0, "cat", 10);
        assert_eq!(related.len(), 3);

        // This depends on the model, but searching for the top two messages with
        // `cat` should only include the messages with cat in them.
        let related = conversation.related(0, "cat", 2);
        assert_eq!(
            related
                .into_iter()
                .filter(|m| m.content().contains("cat"))
                .count(),
            2
        );
    }

    #[test]
    fn test_related_preserves_message_ids() {
        let mut conversation = create_test_conversation();
        conversation.user("First message");
        conversation.assistant("Response");

        let related = conversation.related(0, "message", 10);

        // Each message should have a valid ID
        for message in related {
            let id = match message {
                Message::User { id, .. } | Message::Assistant { id, .. } => id,
                _ => panic!("Unexpected message type"),
            };

            // IDs should have the same conversation timestamp
            assert_eq!(id.conversation, conversation.id);
        }
    }

    #[test]
    fn test_shared_embedder() {
        let embedder: Arc<dyn EmbeddingGenerator> = Arc::new(TestEmbedder {});

        // Create multiple conversations sharing the same embedder
        let mut conversation1 = ConversationBuilder::default()
            .with_embedder(Arc::clone(&embedder))
            .build()
            .unwrap();

        let mut conversation2 = ConversationBuilder::default()
            .with_embedder(Arc::clone(&embedder))
            .build()
            .unwrap();

        // Add messages to both conversations
        conversation1.user("Hello from conversation 1");
        conversation2.user("Hello from conversation 2");

        // Verify both conversations work independently
        assert_eq!(conversation1.messages.len(), 1);
        assert_eq!(conversation2.messages.len(), 1);

        // Verify they can both use the embedder
        let related1 = conversation1.related(0, "hello", 5);
        let related2 = conversation2.related(0, "hello", 5);

        assert_eq!(related1.len(), 1);
        assert_eq!(related2.len(), 1);
    }
}
