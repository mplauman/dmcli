use rustyline::error::ReadlineError;

#[derive(Debug)]
pub enum Error {}

impl From<ReadlineError> for Error {
    fn from(error: ReadlineError) -> Self {
        match error {
            x => panic!("Don't know how to handle {:?}", x),
        }
    }
}
