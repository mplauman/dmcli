use crate::Result;
use libsql::{Builder, Connection};
use tempfile::NamedTempFile;

pub struct Database<const EMBEDDINGS_SIZE: usize> {
    conn: Connection,
    file: NamedTempFile,
}

impl<const EMBEDDINGS_SIZE: usize> Database<EMBEDDINGS_SIZE> {
    pub async fn new() -> Result<Self> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_string_lossy().to_string();

        let db = Builder::new_local(&path).build().await?;
        let conn = db.connect()?;

        conn.execute(
            &format!(
                "CREATE TABLE IF NOT EXISTS embeddings (
                    text TEXT NOT NULL,
                    embedding F32_BLOB({EMBEDDINGS_SIZE})
                )"
            ),
            (),
        )
        .await?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS embeddings_embedding_idx ON embeddings (libsql_vector_idx(embedding))",
            (),
        ).await?;

        Ok(Database { conn, file })
    }

    pub async fn find_similar(
        &self,
        embeddings: &[f32; EMBEDDINGS_SIZE],
        max_results: u64,
    ) -> Result<Vec<String>> {
        let mut rows = self
            .conn
            .query(
                "SELECT * FROM embeddings ORDER BY vector_distance_cos(embedding, ?) ASC LIMIT ?",
                libsql::params![
                    unsafe {
                        let p = embeddings.as_ptr() as *mut u8;
                        let len = embeddings.len() * std::mem::size_of::<f32>();

                        std::slice::from_raw_parts(p, len)
                    },
                    max_results,
                ],
            )
            .await?;

        let mut results = Vec::new();
        while let Some(_row) = rows.next().await? {
            results.push("hello".to_string());
        }

        Ok(results)
    }

    pub async fn insert(&self, _embeddings: &[f32; EMBEDDINGS_SIZE]) -> Result<()> {
        Ok(())
    }
}

impl<const CHUNK_SIZE: usize> std::fmt::Display for Database<CHUNK_SIZE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Database {{ file: {} }}", self.file.path().display())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn create_database() {
        let db = Database::<1024>::new().await.unwrap();
        assert!(db.file.path().exists());
    }

    #[tokio::test]
    async fn search_empty_database() {
        let db = Database::<5>::new().await.unwrap();
        let embeddings: [f32; 5] = [0.0, 0.0, 0.0, 0.0, 0.0];
        let results = db.find_similar(&embeddings, 10).await.unwrap();

        assert!(results.is_empty());
    }
}
