use crate::result::Result;

pub fn index(path: &str, sync: bool) -> Result<IndexStatus> {
    if sync {
        return Ok(IndexStatus::Complete(path.to_string()));
    }

    return Ok(IndexStatus::InProgress(path.to_string()));
}

pub enum IndexStatus {
    Complete(String),
    InProgress(String),
}
