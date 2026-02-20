use crate::Result;
use libsql::{Builder, Connection};
use std::path::Path;
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

        let init_sql = format!(include_str!("init.sql"), EMBEDDINGS_SIZE);
        conn.execute(&init_sql, ()).await?;

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
                include_str!("select_similar_text.sql"),
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
        while let Some(row) = rows.next().await? {
            results.push(row.get_str(0)?.to_string());
        }

        Ok(results)
    }

    pub async fn insert_doc_chunk(
        &self,
        _embedding: &[f32; EMBEDDINGS_SIZE],
        _content: &str,
        _path: &Path,
        _hash: &str,
        _section: &str,
    ) -> Result<()> {
        Ok(())
    }

    pub async fn insert(&self, _embedding: &[f32; EMBEDDINGS_SIZE], text: &str) -> Result<()> {
        self.conn
            .execute(
                include_str!("insert_embedding.sql"),
                libsql::params![
                    unsafe {
                        let p = _embedding.as_ptr() as *mut u8;
                        let len = _embedding.len() * std::mem::size_of::<f32>();

                        std::slice::from_raw_parts(p, len)
                    },
                    text
                ],
            )
            .await?;

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

    #[tokio::test]
    async fn insert_and_find() {
        let db = Database::<5>::new().await.unwrap();
        let embeddings: [f32; 5] = [0.0, 0.0, 0.0, 0.0, 0.0];

        db.insert(&embeddings, "hello").await.unwrap();
        let results = db.find_similar(&embeddings, 10).await.unwrap();

        assert_eq!(results, vec!["hello".to_string()]);
    }
}
