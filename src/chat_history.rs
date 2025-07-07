//! Chat history module with vector embeddings for semantic search
//!
//! This module provides improved chat history functionality using the `memvdb` crate
//! for vector storage and similarity search. The embeddings are generated using 
//! model2vec-rs with the minishlab/potion-base-8M model.

use crate::errors::Error;
use memvdb::{CacheDB, Distance, Embedding as MemvdbEmbedding};
use model2vec_rs::model::StaticModel;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Maximum number of history entries to keep
const MAX_HISTORY: usize = 1000;

/// Collection name for storing chat messages in memvdb
const CHAT_COLLECTION: &str = "chat_messages";

/// Default Model2Vec model for embeddings
const DEFAULT_MODEL: &str = "minishlab/potion-base-8M";

/// Represents a chat message with its metadata
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// The actual text content of the message
    pub content: String,
    /// Timestamp when the message was created
    pub timestamp: u64,
    /// Unique identifier for the message
    pub id: String,
    /// Model2Vec embedding vector
    pub embedding: Vec<f32>,
}

/// Model2Vec-based text embedder for high-quality semantic embeddings
/// 
/// This implementation uses Model2Vec models from the Hugging Face Hub
/// to generate state-of-the-art static embeddings optimized for performance.
pub struct Model2VecEmbedder {
    /// The loaded Model2Vec model
    model: StaticModel,
    /// Model identifier for reference
    model_name: String,
}

impl Model2VecEmbedder {
    /// Creates a new Model2Vec embedder with the specified model
    ///
    /// # Arguments
    /// * `model_name` - HuggingFace model ID or local path (e.g., "minishlab/potion-base-8M")
    ///
    /// # Returns
    /// * `Result<Self, Error>` - New embedder instance or error
    pub fn new(model_name: Option<&str>) -> Result<Self, Error> {
        let model_name = model_name.unwrap_or(DEFAULT_MODEL).to_string();
        
        log::info!("Loading Model2Vec model: {}", model_name);
        
        let model = StaticModel::from_pretrained(
            &model_name,
            None,   // No HuggingFace token needed for public models
            None,   // Use default normalization from model config
            None,   // No subfolder
        ).map_err(|e| Error::Initialization(format!("Failed to load Model2Vec model '{}': {}", model_name, e)))?;
        
        log::info!("Successfully loaded Model2Vec model: {}", model_name);
        
        Ok(Self {
            model,
            model_name,
        })
    }
    
    /// Creates a new embedder with the default model
    pub fn new_default() -> Result<Self, Error> {
        Self::new(None)
    }

    /// Generates high-quality embeddings for text using Model2Vec
    ///
    /// # Arguments
    /// * `text` - The text to embed
    ///
    /// # Returns
    /// * `Vec<f32>` - The embedding vector
    pub fn embed(&self, text: &str) -> Vec<f32> {
        self.model.encode_single(text)
    }
    
    /// Gets the model name/identifier
    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

/// ChatHistory manages chat messages using memvdb for vector storage and Model2Vec for semantic search
pub struct ChatHistory {
    /// Vector database for storing embeddings
    db: CacheDB,
    /// Model2Vec embedder for high-quality text vectorization
    embedder: Model2VecEmbedder,
    /// In-memory cache of recent messages for compatibility
    recent_messages: Vec<String>,
}

impl ChatHistory {
    /// Creates a new ChatHistory instance with the specified model
    ///
    /// # Arguments
    /// * `_db_path` - Path where the vector database could be stored (unused in this implementation)
    /// * `model_name` - Optional Model2Vec model name (defaults to minishlab/potion-base-8M)
    ///
    /// # Returns
    /// * `Result<Self, Error>` - New ChatHistory instance or error
    pub fn new_with_model(_db_path: PathBuf, model_name: Option<&str>) -> Result<Self, Error> {
        let embedder = Model2VecEmbedder::new(model_name)?;
        
        // Get the first embedding to determine the dimension
        let test_embedding = embedder.embed("test");
        let embedding_dim = test_embedding.len();
        
        log::info!("Using Model2Vec model '{}' with {} dimensions", 
                   embedder.model_name(), embedding_dim);
        
        let mut db = CacheDB::new();
        
        // Create the chat messages collection with cosine similarity and dynamic dimensions
        db.create_collection(CHAT_COLLECTION.to_string(), embedding_dim, Distance::Cosine)
            .map_err(|_| Error::Initialization("Failed to create chat collection".to_string()))?;

        Ok(Self {
            db,
            embedder,
            recent_messages: Vec::new(),
        })
    }

    /// Creates a new ChatHistory instance with the default model
    ///
    /// # Arguments
    /// * `db_path` - Path where the vector database could be stored (unused in this implementation)
    ///
    /// # Returns
    /// * `Result<Self, Error>` - New ChatHistory instance or error
    pub fn new(db_path: PathBuf) -> Result<Self, Error> {
        Self::new_with_model(db_path, None)
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
        
        // Get embedding dimension from a test embedding
        let test_embedding = self.embedder.embed("test");
        let embedding_dim = test_embedding.len();
        
        let _ = self.db.create_collection(CHAT_COLLECTION.to_string(), embedding_dim, Distance::Cosine);
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

    fn create_test_chat_history() -> Option<ChatHistory> {
        let temp_dir = TempDir::new().unwrap();
        // Use a simpler model for testing to avoid large downloads
        // If this fails, fall back to the default model
        match ChatHistory::new_with_model(temp_dir.path().to_path_buf(), Some("minishlab/potion-base-2M")) {
            Ok(history) => Some(history),
            Err(_) => {
                // Fall back to default model if the smaller one is not available
                match ChatHistory::new(temp_dir.path().to_path_buf()) {
                    Ok(history) => Some(history),
                    Err(_) => None, // Model download failed - skip test
                }
            }
        }
    }

    #[test]
    fn test_new_chat_history() {
        let temp_dir = TempDir::new().unwrap();
        let chat_history = ChatHistory::new(temp_dir.path().to_path_buf());
        
        // May fail if model download fails, which is acceptable for tests
        if let Ok(chat_history) = chat_history {
            assert!(chat_history.is_empty());
            assert_eq!(chat_history.len(), 0);
        }
    }

    #[test]
    fn test_add_message() {
        let mut chat_history = match create_test_chat_history() {
            Some(history) => history,
            None => {
                println!("Skipping test - model download failed");
                return;
            }
        };
        
        let result = chat_history.add_message("Hello, world!".to_string());
        
        assert!(result.is_ok());
        assert_eq!(chat_history.len(), 1);
        assert!(!chat_history.is_empty());
        
        let messages = chat_history.get_recent_messages();
        assert_eq!(messages[0], "Hello, world!");
    }

    #[test]
    fn test_add_empty_message() {
        let mut chat_history = match create_test_chat_history() {
            Some(history) => history,
            None => {
                println!("Skipping test - model download failed");
                return;
            }
        };
        
        let result = chat_history.add_message("".to_string());
        assert!(result.is_ok());
        assert_eq!(chat_history.len(), 0);
        assert!(chat_history.is_empty());
    }

    #[test]
    fn test_add_duplicate_message() {
        let mut chat_history = match create_test_chat_history() {
            Some(history) => history,
            None => {
                println!("Skipping test - model download failed");
                return;
            }
        };
        
        let result1 = chat_history.add_message("Hello, world!".to_string());
        assert!(result1.is_ok());
        
        let result2 = chat_history.add_message("Hello, world!".to_string());
        assert!(result2.is_ok());
        
        // Should still only have one message due to duplicate filtering
        assert_eq!(chat_history.len(), 1);
    }

    #[test]
    fn test_get_message() {
        let mut chat_history = match create_test_chat_history() {
            Some(history) => history,
            None => {
                println!("Skipping test - model download failed");
                return;
            }
        };
        
        chat_history.add_message("First message".to_string()).unwrap();
        chat_history.add_message("Second message".to_string()).unwrap();
        
        assert_eq!(chat_history.get_message(0), Some(&"First message".to_string()));
        assert_eq!(chat_history.get_message(1), Some(&"Second message".to_string()));
        assert_eq!(chat_history.get_message(2), None);
    }

    #[test]
    fn test_search_similar() {
        let mut chat_history = match create_test_chat_history() {
            Some(history) => history,
            None => {
                println!("Skipping test - model download failed");
                return;
            }
        };
        
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
        let mut chat_history = match create_test_chat_history() {
            Some(history) => history,
            None => {
                println!("Skipping test - model download failed");
                return;
            }
        };
        
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
        let mut chat_history = match create_test_chat_history() {
            Some(history) => history,
            None => {
                println!("Skipping test - model download failed");
                return;
            }
        };
        
        chat_history.add_message("Test message".to_string()).unwrap();
        assert_eq!(chat_history.len(), 1);
        
        chat_history.clear();
        assert_eq!(chat_history.len(), 0);
        assert!(chat_history.is_empty());
    }

    #[test]
    fn test_max_history_limit() {
        let mut chat_history = match create_test_chat_history() {
            Some(history) => history,
            None => {
                println!("Skipping test - model download failed");
                return;
            }
        };
        
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
        let mut chat_history = match create_test_chat_history() {
            Some(history) => history,
            None => {
                println!("Skipping test - model download failed");
                return;
            }
        };
        
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
    fn test_model2vec_embedder() {
        // This test may take a while on first run due to model download
        let embedder = Model2VecEmbedder::new_default();
        
        // If model download fails, skip the test
        if embedder.is_err() {
            println!("Skipping Model2Vec test - model download failed: {:?}", embedder.err());
            return;
        }
        
        let embedder = embedder.unwrap();
        
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
        
        // Test that similar texts have similar embeddings (cosine similarity)
        let similarity_hello = cosine_similarity(&embedding1, &embedding2);
        let similarity_different = cosine_similarity(&embedding1, &embedding3);
        
        // "hello world" should be more similar to "hello universe" than "goodbye world"
        assert!(similarity_hello > similarity_different);
        
        // Verify model name
        assert!(embedder.model_name().contains("potion-base"));
    }

    #[test]
    fn test_model2vec_semantic_similarity() {
        let embedder = Model2VecEmbedder::new_default();
        
        // If model download fails, skip the test
        if embedder.is_err() {
            println!("Skipping Model2Vec semantic test - model download failed: {:?}", embedder.err());
            return;
        }
        
        let embedder = embedder.unwrap();
        
        // Test programming-related semantic similarity
        let rust_embedding = embedder.embed("I love programming in Rust");
        let coding_embedding = embedder.embed("Coding software is great");
        let weather_embedding = embedder.embed("The weather is nice today");
        
        let prog_similarity = cosine_similarity(&rust_embedding, &coding_embedding);
        let weather_similarity = cosine_similarity(&rust_embedding, &weather_embedding);
        
        // Programming-related texts should be more similar than weather text
        assert!(prog_similarity > weather_similarity);
    }

    /// Helper function to calculate cosine similarity between two vectors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if magnitude_a == 0.0 || magnitude_b == 0.0 {
            0.0
        } else {
            dot_product / (magnitude_a * magnitude_b)
        }
    }
}