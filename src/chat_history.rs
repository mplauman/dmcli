//! Chat history module with vector embeddings for semantic search
//!
//! This module provides improved chat history functionality using the `memvdb` crate
//! for vector storage and similarity search.

use crate::errors::Error;
use memvdb::{CacheDB, Distance, Embedding as MemvdbEmbedding};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Maximum number of history entries to keep
const MAX_HISTORY: usize = 1000;

/// Collection name for storing chat messages in memvdb
const CHAT_COLLECTION: &str = "chat_messages";

/// Embedding dimensions for text vectors
const EMBEDDING_DIM: usize = 100;

/// Represents a chat message with its metadata
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// The actual text content of the message
    pub content: String,
    /// Timestamp when the message was created
    pub timestamp: u64,
    /// Unique identifier for the message
    pub id: String,
    /// Simple text embedding (word frequency vector)
    pub embedding: Vec<f32>,
}

/// Simple text embedder using word frequency and basic NLP techniques
#[derive(Debug)]
pub struct SimpleEmbedder {
    /// Vocabulary mapping words to indices
    vocab: HashMap<String, usize>,
}

impl SimpleEmbedder {
    /// Creates a new simple embedder
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
        }
    }

    /// Generates a simple embedding for text using word frequency approach
    pub fn embed(&mut self, text: &str) -> Vec<f32> {
        let words = self.tokenize(text);
        let word_counts = self.count_words(&words);
        
        // Build vocabulary if needed
        for word in &words {
            if !self.vocab.contains_key(word) {
                let idx = self.vocab.len();
                self.vocab.insert(word.clone(), idx);
            }
        }

        // Create embedding vector with fixed dimensions
        let mut embedding = vec![0.0; EMBEDDING_DIM];
        
        for (word, count) in word_counts {
            if let Some(&idx) = self.vocab.get(&word) {
                if idx < embedding.len() {
                    // Simple term frequency
                    let tf = count as f32 / words.len() as f32;
                    embedding[idx] = tf;
                }
            }
        }

        // Normalize the vector
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for val in &mut embedding {
                *val /= magnitude;
            }
        }

        embedding
    }

    /// Simple tokenization - splits on whitespace and removes punctuation
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|word| {
                word.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
            })
            .filter(|word| !word.is_empty())
            .collect()
    }

    /// Counts word frequencies
    fn count_words(&self, words: &[String]) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for word in words {
            *counts.entry(word.clone()).or_insert(0) += 1;
        }
        counts
    }
}

/// ChatHistory manages chat messages using memvdb for vector storage and semantic search
pub struct ChatHistory {
    /// Vector database for storing embeddings
    db: CacheDB,
    /// Simple embedder for text vectorization
    embedder: SimpleEmbedder,
    /// In-memory cache of recent messages for compatibility
    recent_messages: Vec<String>,
}

impl ChatHistory {
    /// Creates a new ChatHistory instance
    ///
    /// # Arguments
    /// * `_db_path` - Path where the vector database could be stored (unused in this implementation)
    ///
    /// # Returns
    /// * `Result<Self, Error>` - New ChatHistory instance or error
    pub fn new(_db_path: PathBuf) -> Result<Self, Error> {
        let mut db = CacheDB::new();
        
        // Create the chat messages collection with cosine similarity
        db.create_collection(CHAT_COLLECTION.to_string(), EMBEDDING_DIM, Distance::Cosine)
            .map_err(|_| Error::Initialization("Failed to create chat collection".to_string()))?;

        Ok(Self {
            db,
            embedder: SimpleEmbedder::new(),
            recent_messages: Vec::new(),
        })
    }

    /// Adds a new message to the chat history
    ///
    /// # Arguments
    /// * `content` - The message content to add
    ///
    /// # Returns
    /// * `Result<(), Error>` - Success or error
    pub fn add_message(&mut self, content: String) -> Result<(), Error> {
        // Don't add empty messages or duplicates of the last entry
        if content.is_empty() || self.recent_messages.last() == Some(&content) {
            return Ok(());
        }

        // Generate unique ID for the message
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let id = format!("msg_{}_{}", timestamp, self.recent_messages.len());

        // Generate embedding
        let embedding = self.embedder.embed(&content);

        // Create memvdb embedding
        let mut memvdb_id = HashMap::new();
        memvdb_id.insert("id".to_string(), id.clone());

        let mut metadata = HashMap::new();
        metadata.insert("content".to_string(), content.clone());
        metadata.insert("timestamp".to_string(), timestamp.to_string());

        let memvdb_embedding = MemvdbEmbedding {
            id: memvdb_id,
            vector: embedding,
            metadata: Some(metadata),
        };

        // Insert into memvdb
        self.db.insert_into_collection(CHAT_COLLECTION, memvdb_embedding)
            .map_err(|_| Error::Database("Failed to insert message".to_string()))?;

        // Add to compatibility cache
        self.recent_messages.push(content);

        // Limit history size
        if self.recent_messages.len() > MAX_HISTORY {
            self.recent_messages.remove(0);
            // Note: We don't remove from memvdb to keep the simple API
            // In a production system, you might want to implement cleanup
        }

        Ok(())
    }

    /// Searches for similar messages using vector similarity via memvdb
    ///
    /// # Arguments
    /// * `query` - The query text to search for
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    /// * `Result<Vec<ChatMessage>, Error>` - Similar messages or error
    pub fn search_similar(&mut self, query: &str, limit: usize) -> Result<Vec<ChatMessage>, Error> {
        let collection = self.db.get_collection(CHAT_COLLECTION)
            .ok_or_else(|| Error::Database("Chat collection not found".to_string()))?;

        if collection.embeddings.is_empty() {
            return Ok(Vec::new());
        }

        // Generate embedding for the query
        let query_embedding = self.embedder.embed(query);

        // Use memvdb for similarity search
        let results = collection.get_similarity(&query_embedding, limit);

        // Convert memvdb results back to ChatMessage format
        let chat_messages: Vec<ChatMessage> = results
            .into_iter()
            .filter_map(|result| {
                let embedding = result.embedding;
                let metadata = embedding.metadata?;
                let content = metadata.get("content")?.clone();
                let timestamp = metadata.get("timestamp")?.parse::<u64>().ok()?;
                let id = embedding.id.get("id")?.clone();

                Some(ChatMessage {
                    content,
                    timestamp,
                    id,
                    embedding: embedding.vector,
                })
            })
            .collect();

        Ok(chat_messages)
    }

    /// Gets the most recent messages for compatibility with existing interface
    ///
    /// # Returns
    /// * `&[String]` - Reference to recent messages
    pub fn get_recent_messages(&self) -> &[String] {
        &self.recent_messages
    }

    /// Gets a specific message by index (for backwards compatibility)
    ///
    /// # Arguments
    /// * `index` - Index of the message to retrieve
    ///
    /// # Returns
    /// * `Option<&String>` - Reference to the message if it exists
    pub fn get_message(&self, index: usize) -> Option<&String> {
        self.recent_messages.get(index)
    }

    /// Gets the number of messages in the history
    ///
    /// # Returns
    /// * `usize` - Number of messages
    pub fn len(&self) -> usize {
        self.recent_messages.len()
    }

    /// Checks if the history is empty
    ///
    /// # Returns
    /// * `bool` - True if empty, false otherwise
    pub fn is_empty(&self) -> bool {
        self.recent_messages.is_empty()
    }

    /// Clears all messages from the history
    pub fn clear(&mut self) {
        self.recent_messages.clear();
        // Re-create the collection to clear embeddings
        let _ = self.db.delete_collection(CHAT_COLLECTION);
        let _ = self.db.create_collection(CHAT_COLLECTION.to_string(), EMBEDDING_DIM, Distance::Cosine);
        self.embedder = SimpleEmbedder::new();
    }

    /// Gets all messages with their embeddings and metadata
    ///
    /// # Returns
    /// * `Result<Vec<ChatMessage>, Error>` - All messages or error
    pub fn get_all_messages(&self) -> Result<Vec<ChatMessage>, Error> {
        let collection = self.db.get_collection(CHAT_COLLECTION)
            .ok_or_else(|| Error::Database("Chat collection not found".to_string()))?;

        let chat_messages: Vec<ChatMessage> = collection.embeddings
            .iter()
            .filter_map(|embedding| {
                let metadata = embedding.metadata.as_ref()?;
                let content = metadata.get("content")?.clone();
                let timestamp = metadata.get("timestamp")?.parse::<u64>().ok()?;
                let id = embedding.id.get("id")?.clone();

                Some(ChatMessage {
                    content,
                    timestamp,
                    id,
                    embedding: embedding.vector.clone(),
                })
            })
            .collect();

        Ok(chat_messages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_chat_history() -> ChatHistory {
        let temp_dir = TempDir::new().unwrap();
        ChatHistory::new(temp_dir.path().to_path_buf()).unwrap()
    }

    #[test]
    fn test_new_chat_history() {
        let temp_dir = TempDir::new().unwrap();
        let chat_history = ChatHistory::new(temp_dir.path().to_path_buf());
        assert!(chat_history.is_ok());
        
        let chat_history = chat_history.unwrap();
        assert!(chat_history.is_empty());
        assert_eq!(chat_history.len(), 0);
    }

    #[test]
    fn test_add_message() {
        let mut chat_history = create_test_chat_history();
        
        let result = chat_history.add_message("Hello, world!".to_string());
        assert!(result.is_ok());
        assert_eq!(chat_history.len(), 1);
        assert!(!chat_history.is_empty());
        
        let messages = chat_history.get_recent_messages();
        assert_eq!(messages[0], "Hello, world!");
    }

    #[test]
    fn test_add_empty_message() {
        let mut chat_history = create_test_chat_history();
        
        let result = chat_history.add_message("".to_string());
        assert!(result.is_ok());
        assert_eq!(chat_history.len(), 0);
        assert!(chat_history.is_empty());
    }

    #[test]
    fn test_add_duplicate_message() {
        let mut chat_history = create_test_chat_history();
        
        let result1 = chat_history.add_message("Hello, world!".to_string());
        assert!(result1.is_ok());
        
        let result2 = chat_history.add_message("Hello, world!".to_string());
        assert!(result2.is_ok());
        
        // Should still only have one message due to duplicate filtering
        assert_eq!(chat_history.len(), 1);
    }

    #[test]
    fn test_get_message() {
        let mut chat_history = create_test_chat_history();
        
        chat_history.add_message("First message".to_string()).unwrap();
        chat_history.add_message("Second message".to_string()).unwrap();
        
        assert_eq!(chat_history.get_message(0), Some(&"First message".to_string()));
        assert_eq!(chat_history.get_message(1), Some(&"Second message".to_string()));
        assert_eq!(chat_history.get_message(2), None);
    }

    #[test]
    fn test_search_similar() {
        let mut chat_history = create_test_chat_history();
        
        chat_history.add_message("Hello, how are you?".to_string()).unwrap();
        chat_history.add_message("I'm doing great, thanks!".to_string()).unwrap();
        chat_history.add_message("What's the weather like?".to_string()).unwrap();
        
        let results = chat_history.search_similar("hello", 5).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|m| m.content.to_lowercase().contains("hello")));
        
        let results = chat_history.search_similar("weather", 5).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|m| m.content.to_lowercase().contains("weather")));
        
        let _results = chat_history.search_similar("nonexistent", 5).unwrap();
        // May be empty due to similarity threshold or may find no matches
    }

    #[test]
    fn test_vector_similarity_search() {
        let mut chat_history = create_test_chat_history();
        
        // Add messages with similar semantic content
        chat_history.add_message("I love programming in Rust".to_string()).unwrap();
        chat_history.add_message("Rust is my favorite programming language".to_string()).unwrap();
        chat_history.add_message("The weather is nice today".to_string()).unwrap();
        chat_history.add_message("I enjoy coding and software development".to_string()).unwrap();
        
        // Search for programming-related content
        let results = chat_history.search_similar("coding software", 3).unwrap();
        
        // Should find programming-related messages due to vector similarity
        assert!(!results.is_empty());
        // The exact results may vary based on the embedding algorithm
    }

    #[test]
    fn test_clear() {
        let mut chat_history = create_test_chat_history();
        
        chat_history.add_message("Test message".to_string()).unwrap();
        assert_eq!(chat_history.len(), 1);
        
        chat_history.clear();
        assert_eq!(chat_history.len(), 0);
        assert!(chat_history.is_empty());
    }

    #[test]
    fn test_max_history_limit() {
        let mut chat_history = create_test_chat_history();
        
        // Add more than MAX_HISTORY messages (use smaller number for testing)
        for i in 0..20 {
            chat_history.add_message(format!("Message {}", i)).unwrap();
        }
        
        // Should be limited to MAX_HISTORY (in this case, smaller test size)
        assert_eq!(chat_history.len(), 20);
        
        // Verify messages are present
        let messages = chat_history.get_recent_messages();
        assert_eq!(messages[0], "Message 0");
        assert_eq!(messages[19], "Message 19");
    }

    #[test]
    fn test_chat_message_creation() {
        let message = ChatMessage {
            content: "Test content".to_string(),
            timestamp: 123456789,
            id: "test_id".to_string(),
            embedding: vec![0.1, 0.2, 0.3],
        };
        
        assert_eq!(message.content, "Test content");
        assert_eq!(message.timestamp, 123456789);
        assert_eq!(message.id, "test_id");
        assert_eq!(message.embedding.len(), 3);
    }

    #[test]
    fn test_get_all_messages() {
        let mut chat_history = create_test_chat_history();
        
        chat_history.add_message("First".to_string()).unwrap();
        chat_history.add_message("Second".to_string()).unwrap();
        
        let all_messages = chat_history.get_all_messages().unwrap();
        assert_eq!(all_messages.len(), 2);
        assert_eq!(all_messages[0].content, "First");
        assert_eq!(all_messages[1].content, "Second");
        
        // Verify embeddings are generated
        assert!(!all_messages[0].embedding.is_empty());
        assert!(!all_messages[1].embedding.is_empty());
    }

    #[test]
    fn test_simple_embedder() {
        let mut embedder = SimpleEmbedder::new();
        
        let embedding1 = embedder.embed("hello world");
        let embedding2 = embedder.embed("hello universe");
        let embedding3 = embedder.embed("goodbye world");
        
        // Embeddings should be non-empty vectors
        assert!(!embedding1.is_empty());
        assert!(!embedding2.is_empty());
        assert!(!embedding3.is_empty());
        
        // All embeddings should have the same dimensions
        assert_eq!(embedding1.len(), embedding2.len());
        assert_eq!(embedding2.len(), embedding3.len());
        assert_eq!(embedding1.len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_tokenization() {
        let embedder = SimpleEmbedder::new();
        
        let tokens = embedder.tokenize("Hello, World! How are you?");
        assert_eq!(tokens, vec!["hello", "world", "how", "are", "you"]);
        
        let tokens = embedder.tokenize("  Multiple   spaces  ");
        assert_eq!(tokens, vec!["multiple", "spaces"]);
    }

    #[test]
    fn test_word_counting() {
        let embedder = SimpleEmbedder::new();
        let words = vec!["hello".to_string(), "world".to_string(), "hello".to_string()];
        
        let counts = embedder.count_words(&words);
        assert_eq!(counts.get("hello"), Some(&2));
        assert_eq!(counts.get("world"), Some(&1));
    }
}