use std::fmt;

/// Errors that can occur in the CLI layer.
#[derive(Debug)]
pub enum Error {
    /// Errors from the library crate.
    Lib(lib::Error),
    /// Configuration file errors.
    Config(String),
    /// I/O errors that occur before the library is involved.
    IO(std::io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Lib(e) => write!(f, "{e}"),
            Error::Config(msg) => write!(f, "configuration error: {msg}"),
            Error::IO(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl From<lib::Error> for Error {
    fn from(e: lib::Error) -> Self {
        Error::Lib(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::IO(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
