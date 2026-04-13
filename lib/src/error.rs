use std::fmt;

#[derive(Debug)]
pub enum Error {
    Roll(String),
    IO(String),
    SQL(String),
    Index(String),
    ModelDownload(String),
    JSON(String),
    AIModel(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Roll(msg) => write!(f, "dice roll error: {msg}"),
            Error::IO(msg) => write!(f, "I/O error: {msg}"),
            Error::SQL(msg) => write!(f, "database error: {msg}"),
            Error::Index(msg) => write!(f, "index error: {msg}"),
            Error::ModelDownload(msg) => write!(f, "model download error: {msg}"),
            Error::JSON(msg) => write!(f, "JSON error: {msg}"),
            Error::AIModel(msg) => write!(f, "AI model error: {msg}"),
        }
    }
}

impl From<caith::RollError> for Error {
    fn from(err: caith::RollError) -> Self {
        use caith::RollError;

        let msg = match err {
            RollError::ParamError(msg) => msg,
            RollError::ParseError(err) => format!("{err}"),
        };

        Error::Roll(msg)
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::IO(format!("{err}"))
    }
}

impl From<hf_hub::api::tokio::ApiError> for Error {
    fn from(err: hf_hub::api::tokio::ApiError) -> Self {
        Error::ModelDownload(err.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Error::JSON(value.to_string())
    }
}

impl From<candle_core::Error> for Error {
    fn from(value: candle_core::Error) -> Self {
        Error::AIModel(value.to_string())
    }
}

impl From<libsql::Error> for Error {
    fn from(err: libsql::Error) -> Self {
        Error::SQL(format!("{err}"))
    }
}

impl std::error::Error for Error {}
