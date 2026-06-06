use std::fmt;

#[derive(Debug)]
pub enum Error {
    Roll(String),
    IO(String),
    JSON(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Roll(msg) => write!(f, "dice roll error: {msg}"),
            Error::IO(msg) => write!(f, "I/O error: {msg}"),
            Error::JSON(msg) => write!(f, "JSON error: {msg}"),
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

impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Error::JSON(value.to_string())
    }
}

impl std::error::Error for Error {}
