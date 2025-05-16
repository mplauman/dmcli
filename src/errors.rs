#[derive(Debug)]
pub enum Error {}

impl From<rustyline::error::ReadlineError> for Error {
    fn from(error: rustyline::error::ReadlineError) -> Self {
        panic!("Don't know how to handle {:?}", error);
    }
}

impl From<config::ConfigError> for Error {
    fn from(error: config::ConfigError) -> Self {
        panic!("Don't know how to handle {:?}", error);
    }
}

impl From<reqwest::Error> for Error {
    fn from(error: reqwest::Error) -> Self {
        panic!("Don't know how to handle {:?}", error);
    }
}
