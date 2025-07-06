//! Chat history module with vector embeddings for semantic search
//!
//! This module provides improved chat history functionality using simple text embeddings
//! and the `memvdb` crate for storage and similarity search.

use crate::errors::Error;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Maximum number of history entries to keep
const MAX_HISTORY: usize = 1000;

/// Represents a chat message with its embedding and metadata
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
    /// Inverse document frequency scores
    idf_scores: HashMap<String, f32>,
}

impl SimpleEmbedder {
    /// Creates a new simple embedder
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            idf_scores: HashMap::new(),
        }
    }

    /// Generates a simple embedding for text using TF-IDF style approach
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

        // Create embedding vector
        let mut embedding = vec![0.0; self.vocab.len().max(100)]; // Minimum 100 dimensions
        
        for (word, count) in word_counts {
            if let Some(&idx) = self.vocab.get(&word) {
                if idx < embedding.len() {
                    // Simple TF-IDF inspired score
                    let tf = count as f32 / words.len() as f32;
                    let idf = self.idf_scores.get(&word).unwrap_or(&1.0);
                    embedding[idx] = tf * idf;
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

    /// Updates IDF scores based on a collection of documents
    pub fn update_idf(&mut self, documents: &[String]) {
        let mut doc_frequencies = HashMap::new();
        
        for doc in documents {
            let words: std::collections::HashSet<String> = self.tokenize(doc).into_iter().collect();
            for word in words {
                *doc_frequencies.entry(word).or_insert(0) += 1;
            }
        }

        let total_docs = documents.len() as f32;
        for (word, freq) in doc_frequencies {
            let idf = (total_docs / freq as f32).ln();
            self.idf_scores.insert(word, idf);
        }
    }
}

/// ChatHistory manages chat messages with vector embeddings for semantic search
pub struct ChatHistory {
    /// Path for potential future file storage
    _db_path: PathBuf,
    /// Simple embedder for text vectorization
    embedder: SimpleEmbedder,
    /// In-memory storage of messages with embeddings
    messages: Vec<ChatMessage>,
    /// In-memory cache of recent messages for compatibility
    recent_messages: Vec<String>,
}

impl ChatHistory {
    /// Creates a new ChatHistory instance
    ///
    /// # Arguments
    /// * `db_path` - Path where the vector database could be stored (future use)
    ///
    /// # Returns
    /// * `Result<Self, Error>` - New ChatHistory instance or error
    pub fn new(db_path: PathBuf) -> Result<Self, Error> {
        Ok(Self {
            _db_path: db_path,
            embedder: SimpleEmbedder::new(),
            messages: Vec::new(),
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
        let id = format!("msg_{}_{}", timestamp, self.messages.len());

        // Generate embedding
        let embedding = self.embedder.embed(&content);

        let message = ChatMessage {
            content: content.clone(),
            timestamp,
            id,
            embedding,
        };

        // Add to storage
        self.messages.push(message);
        self.recent_messages.push(content);

        // Limit history size
        if self.recent_messages.len() > MAX_HISTORY {
            self.recent_messages.remove(0);
            self.messages.remove(0);
        }

        // Update IDF scores periodically for better embeddings
        if self.messages.len() % 10 == 0 {
            let texts: Vec<String> = self.messages.iter().map(|m| m.content.clone()).collect();
            self.embedder.update_idf(&texts);
        }

        Ok(())
    }

    /// Calculates cosine similarity between two vectors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        let min_len = a.len().min(b.len());
        for i in 0..min_len {
            dot_product += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a.sqrt() * norm_b.sqrt())
        }
    }

    /// Searches for similar messages using vector similarity
    ///
    /// # Arguments
    /// * `query` - The query text to search for
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    /// * `Result<Vec<ChatMessage>, Error>` - Similar messages or error
    pub fn search_similar(&mut self, query: &str, limit: usize) -> Result<Vec<ChatMessage>, Error> {
        if self.messages.is_empty() {
            return Ok(Vec::new());
        }

        // Generate embedding for the query
        let query_embedding = self.embedder.embed(query);

        // Calculate similarities and collect results
        let mut similarities: Vec<(f32, &ChatMessage)> = self.messages
            .iter()
            .map(|msg| {
                let similarity = self.cosine_similarity(&query_embedding, &msg.embedding);
                (similarity, msg)
            })
            .collect();

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take the top results and clone them
        let results = similarities
            .into_iter()
            .take(limit)
            .filter(|(sim, _)| *sim > 0.1) // Filter out very low similarity scores
            .map(|(_, msg)| msg.clone())
            .collect();

        Ok(results)
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
        self.messages.clear();
        self.embedder = SimpleEmbedder::new();
    }

    /// Gets all messages with their embeddings and metadata
    ///
    /// # Returns
    /// * `&[ChatMessage]` - Reference to all messages
    pub fn get_all_messages(&self) -> &[ChatMessage] {
        &self.messages
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
        
        // Add more than MAX_HISTORY messages
        for i in 0..(MAX_HISTORY + 10) {
            chat_history.add_message(format!("Message {}", i)).unwrap();
        }
        
        // Should be limited to MAX_HISTORY
        assert_eq!(chat_history.len(), MAX_HISTORY);
        
        // First messages should be removed, last messages should remain
        let messages = chat_history.get_recent_messages();
        assert_eq!(messages[0], "Message 10");
        assert_eq!(messages[MAX_HISTORY - 1], format!("Message {}", MAX_HISTORY + 9));
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
        
        let all_messages = chat_history.get_all_messages();
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
    }

    #[test]
    fn test_cosine_similarity() {
        let chat_history = create_test_chat_history();
        
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![1.0, 0.0, 0.0];
        let vec3 = vec![0.0, 1.0, 0.0];
        
        // Identical vectors should have similarity 1.0
        let sim1 = chat_history.cosine_similarity(&vec1, &vec2);
        assert!((sim1 - 1.0).abs() < 0.001);
        
        // Orthogonal vectors should have similarity 0.0
        let sim2 = chat_history.cosine_similarity(&vec1, &vec3);
        assert!(sim2.abs() < 0.001);
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

    #[test]
    fn test_idf_update() {
        let mut embedder = SimpleEmbedder::new();
        
        let documents = vec![
            "hello world".to_string(),
            "hello universe".to_string(),
            "goodbye world".to_string(),
        ];
        
        embedder.update_idf(&documents);
        
        // "hello" appears in 2/3 documents, should have lower IDF than "universe"
        let hello_idf = embedder.idf_scores.get("hello").unwrap_or(&0.0);
        let universe_idf = embedder.idf_scores.get("universe").unwrap_or(&0.0);
        
        assert!(hello_idf < universe_idf);
    }
}