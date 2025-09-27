use crate::errors::Error;
use model2vec_rs::model::StaticModel;
use std::path::PathBuf;

// Include the generated constants from the build script
include!(concat!(env!("OUT_DIR"), "/model_constants.rs"));

/// A reusable embedding generator that can encode text into vector representations
/// and compute similarity between texts.
///
/// # Examples
///
/// ## Basic Usage
///
/// ```rust
/// use dmcli::embeddings::EmbeddingGeneratorBuilder;
///
/// // Create an embedding generator with default settings
/// let generator = EmbeddingGeneratorBuilder::default().build().unwrap();
///
/// // Encode some text
/// let embedding = generator.encode("Hello world");
/// println!("Embedding has {} dimensions", embedding.len());
///
/// // Compare similarity between texts
/// let embed1 = generator.encode("cat");
/// let embed2 = generator.encode("kitten");
/// let similarity = generator.similarity(&embed1, &embed2);
/// println!("Similarity: {}", similarity);
/// ```
pub struct EmbeddingGenerator {
    model: StaticModel,
}

impl EmbeddingGenerator {
    /// Encodes a single text string into a vector embedding
    ///
    /// # Arguments
    ///
    /// * `text` - The text to encode
    ///
    /// # Returns
    ///
    /// A vector of f32 values representing the text embedding
    pub fn encode(&self, text: &str) -> Vec<f32> {
        self.model.encode_single(text)
    }

    /// Computes the cosine similarity between two embeddings
    ///
    /// # Arguments
    ///
    /// * `a` - First embedding vector
    /// * `b` - Second embedding vector
    ///
    /// # Returns
    ///
    /// A similarity score between 0.0 and 1.0, where 1.0 is most similar
    pub fn similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        // For normalized vectors, cosine similarity is just the dot product
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Computes the cosine distance between two embeddings
    ///
    /// # Arguments
    ///
    /// * `a` - First embedding vector
    /// * `b` - Second embedding vector
    ///
    /// # Returns
    ///
    /// A distance score between 0.0 and 2.0, where 0.0 is most similar
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        // Convert similarity to distance (1 - similarity)
        1.0 - self.similarity(a, b)
    }
}

/// Builder for creating EmbeddingGenerator instances with custom configuration
#[derive(Default)]
pub struct EmbeddingGeneratorBuilder {
    cache_dir: Option<PathBuf>,
}

impl EmbeddingGeneratorBuilder {
    /// Sets the cache directory for storing model files
    #[cfg(test)]
    pub fn with_cache_dir<P: Into<PathBuf>>(mut self, cache_dir: P) -> Self {
        self.cache_dir = Some(cache_dir.into());
        self
    }

    /// Builds the EmbeddingGenerator with the configured options
    pub fn build(self) -> Result<EmbeddingGenerator, Error> {
        let folder = self
            .cache_dir
            .unwrap_or_else(|| dirs::cache_dir().expect("cache dir exists").join("dmcli"));

        if !folder.exists() {
            std::fs::create_dir_all(&folder)?;
        }

        // Always ensure model files exist in the cache directory
        let tokenizer_path = folder.join("tokenizer.json");
        let model_path = folder.join("model.safetensors");
        let config_path = folder.join("config.json");

        if !tokenizer_path.exists() {
            std::fs::write(&tokenizer_path, TOKENIZER_BYTES)?;
        }
        if !model_path.exists() {
            std::fs::write(&model_path, MODEL_BYTES)?;
        }
        if !config_path.exists() {
            std::fs::write(&config_path, CONFIG_BYTES)?;
        }

        let model_path = folder.to_string_lossy().into_owned();

        log::info!("Loading Model2Vec model: {model_path}");
        let model = StaticModel::from_pretrained(
            model_path,
            None,       // No HuggingFace token needed for public models
            Some(true), // Ensure normalized for ranking
            None,       // No subfolder
        )
        .map_err(|e| Error::Embedding(format!("{e}")))?;

        Ok(EmbeddingGenerator { model })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_generator_creation() {
        let temp_dir = tempfile::tempdir().unwrap();
        let generator = EmbeddingGeneratorBuilder::default()
            .with_cache_dir(temp_dir.path())
            .build()
            .unwrap();

        let embedding = generator.encode("test text");
        assert!(!embedding.is_empty());
    }

    #[test]
    fn test_encoding_single_text() {
        let temp_dir = tempfile::tempdir().unwrap();
        let generator = EmbeddingGeneratorBuilder::default()
            .with_cache_dir(temp_dir.path())
            .build()
            .unwrap();

        let embedding = generator.encode("Hello world");
        assert!(!embedding.is_empty());

        // Different texts should produce different embeddings
        let embedding2 = generator.encode("Goodbye world");
        assert_ne!(embedding, embedding2);
    }

    #[test]
    fn test_similarity_calculation() {
        let temp_dir = tempfile::tempdir().unwrap();
        let generator = EmbeddingGeneratorBuilder::default()
            .with_cache_dir(temp_dir.path())
            .build()
            .unwrap();

        let embedding1 = generator.encode("cat");
        let embedding2 = generator.encode("kitten");
        let embedding3 = generator.encode("car");

        let sim_cat_kitten = generator.similarity(&embedding1, &embedding2);
        let sim_cat_car = generator.similarity(&embedding1, &embedding3);

        // Cat and kitten should be more similar than cat and car
        assert!(sim_cat_kitten > sim_cat_car);
    }

    #[test]
    fn test_distance_calculation() {
        let temp_dir = tempfile::tempdir().unwrap();
        let generator = EmbeddingGeneratorBuilder::default()
            .with_cache_dir(temp_dir.path())
            .build()
            .unwrap();

        let embedding1 = generator.encode("dog");
        let embedding2 = generator.encode("puppy");
        let embedding3 = generator.encode("airplane");

        let dist_dog_puppy = generator.distance(&embedding1, &embedding2);
        let dist_dog_airplane = generator.distance(&embedding1, &embedding3);

        // Dog and puppy should be closer (smaller distance) than dog and airplane
        assert!(dist_dog_puppy < dist_dog_airplane);
    }

    #[test]
    fn test_builder_pattern() {
        let temp_dir = tempfile::tempdir().unwrap();
        let generator = EmbeddingGeneratorBuilder::default()
            .with_cache_dir(temp_dir.path())
            .build()
            .unwrap();

        let embedding = generator.encode("test");
        assert!(!embedding.is_empty());
    }

    #[test]
    fn test_similarity_is_symmetric() {
        let temp_dir = tempfile::tempdir().unwrap();
        let generator = EmbeddingGeneratorBuilder::default()
            .with_cache_dir(temp_dir.path())
            .build()
            .unwrap();

        let embedding1 = generator.encode("hello");
        let embedding2 = generator.encode("world");

        let sim1 = generator.similarity(&embedding1, &embedding2);
        let sim2 = generator.similarity(&embedding2, &embedding1);

        assert!((sim1 - sim2).abs() < 1e-6); // Should be equal within floating point precision
    }
}
