use crate::errors::Error;
use tempfile::TempDir;

/// Wrapper around Turso Database for application-specific functionality
pub struct Database {
    conn: libsql::Connection,
    _temp_dir: std::rc::Rc<TempDir>,
}

impl Database {
    /// Create a new database builder
    pub fn builder() -> DatabaseBuilder {
        DatabaseBuilder {}
    }

    /// Create a new temporary file database for testing
    /// This method is only available during testing
    #[cfg(test)]
    pub async fn new() -> Database {
        Self::builder()
            .build()
            .await
            .expect("Failed to create test database")
    }

    /// Create a new connection to this database
    pub fn connect(&self) -> Result<Connection, Error> {
        Ok(Connection {
            inner: self.conn.clone(),
            _tmp: self._temp_dir.clone(),
        })
    }
}

/// Wrapper around Turso Connection for application-specific functionality
pub struct Connection {
    inner: libsql::Connection,
    _tmp: std::rc::Rc<TempDir>,
}

impl Connection {
    /// Execute a SQL statement with parameters
    pub async fn execute(
        &self,
        sql: &str,
        params: impl libsql::params::IntoParams,
    ) -> Result<(), Error> {
        self.inner
            .execute(sql, params)
            .await
            .map(|_| ()) // Discard the row count and return unit
            .map_err(|e| Error::Embedding(format!("Database execution failed: {}", e)))
    }

    pub async fn query(
        &self,
        sql: &str,
        params: impl libsql::params::IntoParams,
    ) -> Result<libsql::Rows, Error> {
        let rows = self
            .inner
            .prepare(sql)
            .await
            .expect("SQL structure is valid")
            .query(params)
            .await
            .expect("SQL query executes correctly");

        Ok(rows)
    }
}

/// Builder for creating Database instances
pub struct DatabaseBuilder {}

impl DatabaseBuilder {
    /// Build the database instance with a temporary file
    pub async fn build(self) -> Result<Database, Error> {
        // Create a temporary directory
        let temp_dir = TempDir::new()
            .map_err(|e| Error::Embedding(format!("Failed to create temp directory: {}", e)))?;

        // Create database file path within the temp directory
        let db_path = temp_dir.path().join("database.sqlite");
        let location = db_path.to_string_lossy().to_string();

        let db = libsql::Builder::new_local(&location)
            .build()
            .await
            .map_err(|e| Error::Embedding(format!("Failed to create database: {}", e)))?;

        // Initialize schema using raw connection
        let conn = db
            .connect()
            .map_err(|e| Error::Embedding(format!("Failed to connect to database: {}", e)))?;

        // Return wrapped database with temp directory reference
        Ok(Database {
            conn,
            _temp_dir: std::rc::Rc::new(temp_dir),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_temp_file_database_cleanup() {
        let temp_path;

        // Create database and capture the temp file path
        {
            let db = Database::new().await;

            // Extract the temp directory path for verification
            temp_path = db._temp_dir.path().to_path_buf();

            // Verify the temp directory exists while database is in scope
            assert!(
                temp_path.exists(),
                "Temp directory should exist while database is in scope"
            );

            // Verify we can connect to the database
            let conn = db
                .connect()
                .expect("Should be able to connect to temp file database");

            // Verify we can execute a simple query
            conn.execute("CREATE TABLE test (id INTEGER)", ())
                .await
                .expect("Should be able to create table in temp file database");
        } // Database goes out of scope here

        // Give a moment for cleanup to occur
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Verify the temp directory is cleaned up after database is dropped
        assert!(
            !temp_path.exists(),
            "Temp directory should be cleaned up when database is dropped"
        );
    }

    #[tokio::test]
    async fn test_builder_creates_temp_file() {
        let db = Database::builder()
            .build()
            .await
            .expect("Should be able to create database with temp file");

        let temp_path = db._temp_dir.path().to_path_buf();
        assert!(temp_path.exists(), "Temp directory should exist");

        // Verify the database file exists within the temp directory
        let db_file = temp_path.join("database.sqlite");
        assert!(
            db_file.exists(),
            "Database file should exist in temp directory"
        );
    }
}
