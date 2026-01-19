#[derive(Debug)]
pub enum Error {
    Roll(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Roll(msg) => write!(f, "{msg}"),
        }
    }
}

impl From<caith::RollError> for Error {
    fn from(err: caith::RollError) -> Self {
        use caith::RollError;

        let msg = match err {
            RollError::ParamError(msg) => format!("Bad dice roll: {msg}"),
            RollError::ParseError(msg) => format!("Bad dice roll: {msg}"),
        };

        Error::Roll(msg)
    }
}
