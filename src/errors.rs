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

impl From<opentelemetry_otlp::ExporterBuildError> for Error {
    fn from(error: opentelemetry_otlp::ExporterBuildError) -> Self {
        panic!("Don't know how to handle {:?}", error);
    }
}

impl From<log::SetLoggerError> for Error {
    fn from(error: log::SetLoggerError) -> Self {
        panic!("Don't know how to handle {:?}", error);
    }
}

impl From<syslog::Error> for Error {
    fn from(error: syslog::Error) -> Self {
        panic!("Don't know how to handle {:?}", error);
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        panic!("Don't know how to handle {:?}", error);
    }
}
