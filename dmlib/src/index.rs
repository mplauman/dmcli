use crate::result::{DmlibResult, Result};

pub fn index(path: &str, sync: bool) -> Result {
    if sync {
        return Ok(DmlibResult::IndexResult(path.to_string()));
    }

    return Ok(DmlibResult::AsyncIndexResult(path.to_string()));
}
