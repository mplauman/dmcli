#[derive(Debug)]
pub enum Error {
    Roll(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Roll(msg) => write!(f, "Bad dice roll: {msg}"),
        }
    }
}

impl From<dmlib::Error> for Error {
    fn from(err: dmlib::Error) -> Self {
        match err {
            dmlib::Error::Roll(msg) => Error::Roll(msg),
        }
    }
}
