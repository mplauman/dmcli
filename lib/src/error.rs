#[derive(Debug)]
pub enum Error {
    Roll(String),
    IO(String),
    SQL(String),
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
        Error::Roll(format!("{err}"))
    }
}

impl From<libsql::Error> for Error {
    fn from(err: libsql::Error) -> Self {
        Error::SQL(format!("{err}"))
    }
}
