use crate::errors::Error;
use model2vec_rs::model::StaticModel;

pub const EMBEDDING_DIMS: usize = 256;
pub type Embedding = [f32; EMBEDDING_DIMS];

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
    #[allow(async_fn_in_trait)]
    async fn encode(&self, text: &str) -> Result<Embedding, Error>;

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
    fn similarity(&self, a: &Embedding, b: &Embedding) -> f32 {
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
    fn distance(&self, a: &Embedding, b: &Embedding) -> f32 {
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
    async fn encode(&self, text: &str) -> Result<Embedding, Error> {
        Ok(self
            .model
            .encode_single(text)
            .try_into()
            .expect("embeddings are the correct size"))
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
    async fn encode(&self, text: &str) -> Result<Embedding, Error> {
        // Simple encoding that just counts vowels and consonants
        let vowels = text.chars().filter(|c| "aeiou".contains(*c)).count() as f32;
        let consonants = text
            .chars()
            .filter(|c| c.is_alphabetic() && !"aeiou".contains(*c))
            .count() as f32;

        let mut result = [0.0; EMBEDDING_DIMS];
        result[0] = vowels;
        result[1] = consonants;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_model2vec_embedding_generator() {
        let generator = Model2VecEmbeddingGeneratorBuilder::default()
            .build()
            .unwrap();

        // Different texts should produce different embeddings
        let embedding = generator.encode("test text").await.unwrap();
        let embedding2 = generator.encode("Goodbye world").await.unwrap();
        assert_ne!(embedding, embedding2);

        test_similarity_calculation(&generator).await;
        test_distance_calculation(&generator).await;
        test_similarity_is_symmetric(&generator).await;
    }

    async fn test_similarity_calculation(generator: &Model2VecEmbeddingGenerator) {
        let embedding1 = generator.encode("cat").await.unwrap();
        let embedding2 = generator.encode("kitten").await.unwrap();
        let embedding3 = generator.encode("car").await.unwrap();

        let sim_cat_kitten = generator.similarity(&embedding1, &embedding2);
        let sim_cat_car = generator.similarity(&embedding1, &embedding3);

        // Cat and kitten should be more similar than cat and car
        assert!(sim_cat_kitten > sim_cat_car);
    }

    async fn test_distance_calculation(generator: &Model2VecEmbeddingGenerator) {
        let embedding1 = generator.encode("dog").await.unwrap();
        let embedding2 = generator.encode("puppy").await.unwrap();
        let embedding3 = generator.encode("airplane").await.unwrap();

        let dist_dog_puppy = generator.distance(&embedding1, &embedding2);
        let dist_dog_airplane = generator.distance(&embedding1, &embedding3);

        // Dog and puppy should be closer (smaller distance) than dog and airplane
        assert!(dist_dog_puppy < dist_dog_airplane);
    }

    async fn test_similarity_is_symmetric(generator: &Model2VecEmbeddingGenerator) {
        let embedding1 = generator.encode("hello").await.unwrap();
        let embedding2 = generator.encode("world").await.unwrap();

        let sim1 = generator.similarity(&embedding1, &embedding2);
        let sim2 = generator.similarity(&embedding2, &embedding1);

        assert!((sim1 - sim2).abs() < 1e-6); // Should be equal within floating point precision
    }
}
