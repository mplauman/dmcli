use crate::error::Error;

pub type Result = std::result::Result<DmlibResult, Error>;

pub enum DmlibResult {
    SingleDiceRoll(i64, Option<String>),
    MultiDiceRoll(Vec<i64>, Option<String>),
}
