use crate::{DmlibResult, Error, Result};
use libsql::{Builder, Connection};
use tempfile::NamedTempFile;

pub struct Database {
    conn: Connection,
    file: NamedTempFile,
}

impl Database {
    pub async fn new() -> core::result::Result<Self, Error> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_string_lossy().to_string();

        let db = Builder::new_local(&path).build().await?;
        let conn = db.connect()?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT NOT NULL,
                    embedding F32_BLOB(64)
                )",
            (),
        )
        .await?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS embeddings_embedding_idx ON embeddings (libsql_vector_idx(embedding))",
            (),
        ).await?;

        Ok(Database { conn, file })
    }

    pub async fn search(&self, _text: &str, max_results: u64) -> Result {
        let mut rows = self
            .conn
            .query(
                "SELECT * FROM embeddings ORDER BY vector_distance_cos(embedding, 3) ASC LIMIT ?",
                libsql::params![
                    // unsafe {
                    //     let p = target.as_ptr() as *mut u8;
                    //     let len = target.len() * std::mem::size_of::<f32>();

                    //     std::slice::from_raw_parts(p, len)
                    // },
                    max_results,
                ],
            )
            .await?;

        let mut results = Vec::new();
        while let Some(_row) = rows.next().await? {
            results.push("hello".to_string());
        }

        Ok(DmlibResult::SearchResult(results))
    }
}

impl std::fmt::Display for Database {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Database {{ file: {} }}", self.file.path().display())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn create_database() {
        let db = Database::new().await.unwrap();
        assert!(db.file.path().exists());
    }

    #[tokio::test]
    async fn search_empty_database() {
        let db = Database::new().await.unwrap();
        let DmlibResult::SearchResult(results) = db.search("", 10).await.unwrap() else {
            panic!()
        };

        assert!(results.is_empty());
    }
}
