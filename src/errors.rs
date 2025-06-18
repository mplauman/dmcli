#[derive(Debug)]
pub enum Error {
    NoToolUses,
    Io(std::io::Error),
    Eof,
    Interrupted,
    WindowResized,
    JsonDeserialization(usize, usize, serde_json::error::Category),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use serde_json::error::Category;

        match self {
            Self::NoToolUses => write!(f, "No tools provided for execution"),
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::Eof => write!(f, "EOF"),
            Self::Interrupted => write!(f, "Interrupted"),
            Self::WindowResized => write!(f, "Window resized"),
            Self::JsonDeserialization(line, column, Category::Io) => write!(
                f,
                "IO failure while deserializing JSON (line {}, col {})",
                line, column
            ),
            Self::JsonDeserialization(line, column, Category::Eof) => write!(
                f,
                "EOF while deserializing JSON (line {}, col {})",
                line, column
            ),
            Self::JsonDeserialization(line, column, Category::Data) => write!(
                f,
                "Data conversion error while deserializing JSON (line {}, col {})",
                line, column
            ),
            Self::JsonDeserialization(line, column, Category::Syntax) => write!(
                f,
                "Syntax error while deserializing JSON (line {}, col {})",
                line, column
            ),
        }
    }
}

impl From<rustyline::error::ReadlineError> for Error {
    fn from(error: rustyline::error::ReadlineError) -> Self {
        use rustyline::error::ReadlineError;

        match error {
            ReadlineError::Io(e) => Self::Io(e),
            ReadlineError::Eof => Self::Eof,
            ReadlineError::Interrupted => Self::Interrupted,
            ReadlineError::WindowResized => Self::WindowResized,
            x => panic!("Unexpected readline error: {}", x),
        }
    }
}

impl From<config::ConfigError> for Error {
    fn from(error: config::ConfigError) -> Self {
        panic!("Don't know how to handle {error:?}");
    }
}

impl From<reqwest::Error> for Error {
    fn from(error: reqwest::Error) -> Self {
        panic!("Don't know how to handle {error:?}");
    }
}

impl From<opentelemetry_otlp::ExporterBuildError> for Error {
    fn from(error: opentelemetry_otlp::ExporterBuildError) -> Self {
        panic!("Don't know how to handle {error:?}");
    }
}

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
        panic!("Don't know how to handle {}", error);
    }
}

impl From<Error> for rmcp::Error {
    fn from(val: Error) -> Self {
        use serde_json::error::Category;

        match val {
            Error::NoToolUses => rmcp::Error::invalid_request("No tool uses found", None),
            Error::Io(e) => rmcp::Error::internal_error(format!("IO error: {}", e), None),
            Error::Eof => rmcp::Error::internal_error("Unhandled EOF during tool usage", None),
            Error::Interrupted => {
                rmcp::Error::internal_error("Interrupted during tool usage", None)
            }
            Error::WindowResized => {
                panic!("Window resize events should not happen during tool use")
            }
            Error::JsonDeserialization(line, column, Category::Io) => rmcp::Error::internal_error(
                format!(
                    "IO failure during tool processing, decoding intermediate JSON. Line {}, col {}.",
                    line, column,
                ),
                None,
            ),
            Error::JsonDeserialization(line, column, Category::Eof) => rmcp::Error::internal_error(
                format!(
                    "Unexpected EOF during tool processing, decoding intermediate JSON. Line {}, col {}.",
                    line, column,
                ),
                None,
            ),
            Error::JsonDeserialization(line, column, Category::Data) => panic!(
                "Bad data conversion during tool use while decoding JSON. Incompatible data on line {}, col {}",
                line, column
            ),
            Error::JsonDeserialization(line, column, Category::Syntax) => panic!(
                "Invalid JSON syntax during tool use while decoding JSON on line {}, column {}",
                line, column
            ),
        }
    }
}
