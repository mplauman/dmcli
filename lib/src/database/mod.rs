use crate::Result;
use libsql::{Builder, Connection};
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;

/// Tracks the backing storage for the database file.
enum DatabaseFile {
    /// A temporary file that is deleted when dropped.
    Temporary(NamedTempFile),
    /// A persistent file at a user-supplied path.
    Persistent(PathBuf),
}

impl DatabaseFile {
    fn path(&self) -> &Path {
        match self {
            DatabaseFile::Temporary(f) => f.path(),
            DatabaseFile::Persistent(p) => p.as_path(),
        }
    }
}

/// A local SQLite vector database backed by either a temporary file or a
/// caller-supplied path.
pub struct Database<const EMBEDDINGS_SIZE: usize> {
    conn: Connection,
    file: DatabaseFile,
}

impl<const EMBEDDINGS_SIZE: usize> Database<EMBEDDINGS_SIZE> {
    /// Create a new database backed by a temporary file that is deleted on drop.
    pub async fn new() -> Result<Self> {
        let file = NamedTempFile::new()?;
        Self::open_file(DatabaseFile::Temporary(file)).await
    }

    /// Open (or create) a database at the given path. The file persists after
    /// the [`Database`] is dropped.
    pub async fn open(path: PathBuf) -> Result<Self> {
        Self::open_file(DatabaseFile::Persistent(path)).await
    }

    async fn open_file(file: DatabaseFile) -> Result<Self> {
        let path = file.path().to_string_lossy().to_string();
        let db = Builder::new_local(&path).build().await?;
        let conn = db.connect()?;

        let init_sql = format!(include_str!("init.sql"), EMBEDDINGS_SIZE);
        conn.execute_batch(&init_sql).await?;

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
        embedding: &[f32; EMBEDDINGS_SIZE],
        content: &str,
        path: &Path,
        hash: &str,
        section: &str,
    ) -> Result<()> {
        self.insert(embedding, content).await?;

        let row_id = self.conn.last_insert_rowid();
        println!("Inserted chunk with id {row_id}");

        self.conn
            .execute(
                include_str!("insert_document_chunks.sql"),
                libsql::params![row_id, path.to_string_lossy(), section, hash],
            )
            .await?;

        Ok(())
    }

    pub async fn insert(&self, embedding: &[f32; EMBEDDINGS_SIZE], text: &str) -> Result<()> {
        self.conn
            .execute(
                include_str!("insert_embedding.sql"),
                libsql::params![
                    unsafe {
                        let p = embedding.as_ptr() as *mut u8;
                        let len = embedding.len() * std::mem::size_of::<f32>();

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
    async fn create_database_at_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");

        let db = Database::<1024>::open(path.clone()).await.unwrap();
        assert!(path.exists());
        drop(db);

        // File must persist after drop.
        assert!(path.exists());
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

    #[tokio::test]
    async fn insert_doc_chunk() {
        let db = Database::<5>::new().await.unwrap();
        let embeddings: [f32; 5] = [0.0, 0.0, 0.0, 0.0, 0.0];

        db.insert_doc_chunk(&embeddings, "hello", Path::new("foo.md"), "hash", "section")
            .await
            .unwrap();

        let results = db.find_similar(&embeddings, 10).await.unwrap();

        assert_eq!(results, vec!["hello".to_string()]);
    }
}
