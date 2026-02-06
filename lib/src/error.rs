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

impl From<hf_hub::api::sync::ApiError> for Error {
    fn from(err: hf_hub::api::sync::ApiError) -> Self {
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
