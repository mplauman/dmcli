pub enum Error {
    Roll(String),
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
