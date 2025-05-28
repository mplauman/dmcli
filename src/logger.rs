use log::{Log, Metadata, Record};

pub struct AggregateLogger {
    loggers: Vec<Box<dyn Log>>,
}

#[derive(Default)]
pub struct AggregateLoggerBuilder {
    loggers: Vec<Box<dyn Log>>,
}

impl AggregateLoggerBuilder {
    pub fn with<T: Log + 'static>(self, logger: T) -> Self {
        let mut loggers = self.loggers;
        loggers.push(Box::new(logger));

        AggregateLoggerBuilder { loggers }
    }

    pub fn build(self) -> AggregateLogger {
        AggregateLogger {
            loggers: self.loggers,
        }
    }
}

impl Log for AggregateLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        self.loggers.iter().any(|logger| logger.enabled(metadata))
    }

    fn log(&self, record: &Record) {
        for logger in self.loggers.iter() {
            logger.log(record);
        }
    }

    fn flush(&self) {
        for logger in self.loggers.iter() {
            logger.flush();
        }
    }
}
