use crate::errors::Error;
use model2vec_rs::model::StaticModel;

pub trait EmbeddingGenerator {
    /// Encodes a single text string into a vector embedding
    ///
    /// # Arguments
    ///
    /// * `text` - The text to encode
    ///
    /// # Returns
    ///
    /// A vector of f32 values representing the text embedding
    fn encode(&self, text: &str) -> Vec<f32>;

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
    fn similarity(&self, a: &[f32], b: &[f32]) -> f32 {
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
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        1.0 - self.similarity(a, b)
    }
}

/// A reusable embedding generator that can encode text into vector representations
/// and compute similarity between texts.
///
/// # Examples
///
/// ## Basic Usage
///
/// ```rust
/// use dmcli::embeddings::Model2VecEmbeddingGeneratorBuilder;
///
/// // Create an embedding generator with default settings
/// let generator = Model2VecEmbeddingGeneratorBuilder::default().build().unwrap();
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
pub struct Model2VecEmbeddingGenerator {
    model: StaticModel,
}

impl EmbeddingGenerator for Model2VecEmbeddingGenerator {
    fn encode(&self, text: &str) -> Vec<f32> {
        self.model.encode_single(text)
    }
}

/// Builder for creating Model2VecEmbeddingGeneratorBuilder instances with custom configuration
pub struct Model2VecEmbeddingGeneratorBuilder {
    repo: String,
    token: Option<String>,
    subfolder: Option<String>,
}

impl Default for Model2VecEmbeddingGeneratorBuilder {
    fn default() -> Self {
        Self {
            repo: "minishlab/potion-base-8M".to_owned(),
            token: None,
            subfolder: None,
        }
    }
}

impl Model2VecEmbeddingGeneratorBuilder {
    /// Sets the HuggingFace repo ID or local path to the model folder
    pub fn repo(mut self, repo: String) -> Self {
        self.repo = repo;
        self
    }

    /// Sets the HuggingFace token for authenticated downloads
    pub fn token(mut self, token: String) -> Self {
        self.token = Some(token);
        self
    }

    /// Sets the subfolder within the repo or path to look for model files
    pub fn subfolder(mut self, subfolder: String) -> Self {
        self.subfolder = Some(subfolder);
        self
    }

    /// Builds the EmbeddingGenerator with the configured options
    pub fn build(self) -> Result<Model2VecEmbeddingGenerator, Error> {
        let model = StaticModel::from_pretrained(
            self.repo,
            self.token.as_deref(),
            None,
            self.subfolder.as_deref(),
        )
        .map_err(|e| Error::Embedding(format!("{e}")))?;

        Ok(Model2VecEmbeddingGenerator { model })
    }
}

/// Mock embedder implementation for testing
#[cfg(test)]
#[derive(Default)]
pub struct TestEmbedder;

#[cfg(test)]
impl EmbeddingGenerator for TestEmbedder {
    fn encode(&self, text: &str) -> Vec<f32> {
        // Simple encoding that just counts vowels and consonants
        let vowels = text.chars().filter(|c| "aeiou".contains(*c)).count() as f32;
        let consonants = text
            .chars()
            .filter(|c| c.is_alphabetic() && !"aeiou".contains(*c))
            .count() as f32;
        vec![vowels, consonants]
    }

    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        // Simple Euclidean distance calculation
        if a.len() != 2 || b.len() != 2 {
            return f32::MAX;
        }
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        (dx * dx + dy * dy).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model2vec_embedding_generator() {
        let generator = Model2VecEmbeddingGeneratorBuilder::default()
            .build()
            .unwrap();

        let embedding = generator.encode("test text");
        assert!(!embedding.is_empty());

        // Different texts should produce different embeddings
        let embedding2 = generator.encode("Goodbye world");
        assert_ne!(embedding, embedding2);
    }

    #[test]
    fn test_similarity_calculation() {
        let generator = TestEmbedder {};

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
        let generator = TestEmbedder {};

        let embedding1 = generator.encode("dog");
        let embedding2 = generator.encode("puppy");
        let embedding3 = generator.encode("airplane");

        let dist_dog_puppy = generator.distance(&embedding1, &embedding2);
        let dist_dog_airplane = generator.distance(&embedding1, &embedding3);

        // Dog and puppy should be closer (smaller distance) than dog and airplane
        assert!(dist_dog_puppy < dist_dog_airplane);
    }

    #[test]
    fn test_similarity_is_symmetric() {
        let generator = TestEmbedder {};

        let embedding1 = generator.encode("hello");
        let embedding2 = generator.encode("world");

        let sim1 = generator.similarity(&embedding1, &embedding2);
        let sim2 = generator.similarity(&embedding2, &embedding1);

        assert!((sim1 - sim2).abs() < 1e-6); // Should be equal within floating point precision
    }
}
