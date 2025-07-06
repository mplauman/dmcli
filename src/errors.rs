#[derive(Debug)]
pub enum Error {
    NoToolUses,
    Io(std::io::Error),
    Eof,
    Interrupted,
    WindowResized,
    JsonDeserialization(usize, usize, serde_json::error::Category),
    InvalidVaultPath(String),
    Http(reqwest::Error),
    Database(String),
    Initialization(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use serde_json::error::Category;

        match self {
            Self::NoToolUses => write!(f, "No tools provided for execution"),
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::Eof => write!(f, "EOF"),
            Self::Interrupted => write!(f, "Interrupted"),
            Self::WindowResized => write!(f, "Window resized"),
            Self::JsonDeserialization(line, column, Category::Io) => write!(
                f,
                "IO failure while deserializing JSON (line {line}, col {column})"
            ),
            Self::JsonDeserialization(line, column, Category::Eof) => write!(
                f,
                "EOF while deserializing JSON (line {line}, col {column})"
            ),
            Self::JsonDeserialization(line, column, Category::Data) => write!(
                f,
                "Data conversion error while deserializing JSON (line {line}, col {column})"
            ),
            Self::JsonDeserialization(line, column, Category::Syntax) => write!(
                f,
                "Syntax error while deserializing JSON (line {line}, col {column})"
            ),
            Self::InvalidVaultPath(path) => write!(
                f,
                "Invalid vault path '{path}': paths must be relative to the vault root, not absolute"
            ),
            Self::Http(e) => write!(f, "HTTP error: {e}"),
            Self::Database(msg) => write!(f, "Database error: {msg}"),
            Self::Initialization(msg) => write!(f, "Initialization error: {msg}"),
        }
    }
}

impl From<std::path::StripPrefixError> for Error {
    fn from(error: std::path::StripPrefixError) -> Self {
        panic!("Don't know how to handle {error}");
    }
}

impl From<config::ConfigError> for Error {
    fn from(error: config::ConfigError) -> Self {
        panic!("Don't know how to handle {error:?}");
    }
}

impl From<reqwest::Error> for Error {
    fn from(error: reqwest::Error) -> Self {
        Self::Http(error)
    }
}

impl From<opentelemetry_otlp::ExporterBuildError> for Error {
    fn from(error: opentelemetry_otlp::ExporterBuildError) -> Self {
        panic!("Don't know how to handle {error:?}");
    }
}

#[cfg(unix)]
impl From<syslog::Error> for Error {
    fn from(error: syslog::Error) -> Self {
        panic!("Don't know how to handle {error:?}");
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self::JsonDeserialization(error.line(), error.column(), error.classify())
    }
}

impl From<rmcp::ServiceError> for Error {
    fn from(error: rmcp::ServiceError) -> Self {
        panic!("Don't know how to handle {error:?}");
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<Error> for rmcp::Error {
    fn from(val: Error) -> Self {
        use serde_json::error::Category;

        match val {
            Error::NoToolUses => rmcp::Error::invalid_request("No tool uses found", None),
            Error::Io(e) => rmcp::Error::internal_error(format!("IO error: {e}"), None),
            Error::Eof => rmcp::Error::internal_error("Unhandled EOF during tool usage", None),
            Error::Interrupted => {
                rmcp::Error::internal_error("Interrupted during tool usage", None)
            }
            Error::WindowResized => {
                panic!("Window resize events should not happen during tool use")
            }
            Error::JsonDeserialization(line, column, Category::Io) => rmcp::Error::internal_error(
                format!(
                    "IO failure during tool processing, decoding intermediate JSON. Line {line}, col {column}."
                ),
                None,
            ),
            Error::JsonDeserialization(line, column, Category::Eof) => rmcp::Error::internal_error(
                format!(
                    "Unexpected EOF during tool processing, decoding intermediate JSON. Line {line}, col {column}."
                ),
                None,
            ),
            Error::JsonDeserialization(line, column, Category::Data) => panic!(
                "Bad data conversion during tool use while decoding JSON. Incompatible data on line {line}, col {column}"
            ),
            Error::JsonDeserialization(line, column, Category::Syntax) => panic!(
                "Invalid JSON syntax during tool use while decoding JSON on line {line}, column {column}"
            ),
            Error::InvalidVaultPath(path) => rmcp::Error::invalid_request(
                format!(
                    "Invalid vault path '{path}': paths must be relative to the vault root, not absolute"
                ),
                None,
            ),
            Error::Http(e) => rmcp::Error::internal_error(format!("HTTP error: {e}"), None),
            Error::Database(msg) => rmcp::Error::internal_error(format!("Database error: {msg}"), None),
            Error::Initialization(msg) => rmcp::Error::internal_error(format!("Initialization error: {msg}"), None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http_error_enum_variant_exists() {
        // Test that Http variant exists and can be matched
        // We create a dummy reqwest error by making an invalid URL request
        let _result = std::panic::catch_unwind(|| {
            // This will create a reqwest error when we try to build an invalid request
            let client = reqwest::Client::new();
            // Use a clearly invalid URL scheme to trigger an error
            client.get("not-a-valid-url-scheme://example.com").build()
        });

        // We don't need the actual error, just testing the enum variant exists
        // and can be handled properly in match statements
        match Error::NoToolUses {
            Error::Http(_) => unreachable!(),
            Error::NoToolUses => {} // This should match
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_error_display_coverage() {
        // Test display implementations for all variants
        let errors = vec![
            Error::NoToolUses,
            Error::Eof,
            Error::Interrupted,
            Error::WindowResized,
            Error::InvalidVaultPath("/test/path".to_string()),
        ];

        for error in errors {
            let display_str = format!("{error}");
            assert!(!display_str.is_empty());
        }
    }

    #[test]
    fn test_invalid_vault_path_error() {
        let error = Error::InvalidVaultPath("/absolute/path".to_string());
        let display_str = format!("{error}");
        assert!(display_str.contains("Invalid vault path"));
        assert!(display_str.contains("absolute"));
        assert!(display_str.contains("relative to the vault root"));
    }

    #[tokio::test]
    async fn test_reqwest_error_integration() {
        // Test that reqwest::Error properly converts to Error::Http
        let client = reqwest::Client::new();

        // Make a request to an invalid URL to generate a real reqwest::Error
        let result = client
            .get("http://this-domain-absolutely-does-not-exist-12345.invalid")
            .send()
            .await;

        // This should fail and give us a reqwest::Error
        if let Err(reqwest_error) = result {
            // Convert to our Error type
            let our_error: Error = reqwest_error.into();

            // Verify it's the Http variant
            match our_error {
                Error::Http(_) => {
                    // Test display
                    let display_str = format!("{our_error}");
                    assert!(display_str.contains("HTTP error:"));

                    // Test conversion to rmcp::Error
                    let rmcp_error: rmcp::Error = our_error.into();
                    let rmcp_str = rmcp_error.to_string();
                    assert!(rmcp_str.contains("HTTP error:"));
                }
                _ => panic!("Expected Error::Http variant"),
            }
        }
    }
}
