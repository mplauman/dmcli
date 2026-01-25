use crate::result::Result;
use std::fs::read_to_string;
use text_splitter::MarkdownSplitter;
use walkdir::{DirEntry, WalkDir};

pub fn index(path: &str, sync: bool) -> Result<IndexStatus> {
    if !sync {
        return Ok(IndexStatus::InProgress(format!("indexing {path}")));
    }

    let walker = WalkDir::new(path)
        .into_iter()
        .filter_entry(|e| {
            let hidden = e
                .file_name()
                .to_str()
                .map(|name| name.starts_with("."))
                .unwrap_or(false);

            !hidden
        })
        .filter_map(|e| e.map(DirEntry::into_path).ok())
        .filter(|e| e.is_file())
        .filter(|e| e.extension().and_then(|e| e.to_str()) == Some("md"));

    let splitter = MarkdownSplitter::new(1014);

    for entry in walker {
        println!("Processing {entry:?}");

        let contents = read_to_string(entry)?;
        let chunks = splitter.chunks(&contents).count();
        println!(" => {chunks} chunks");
    }

    return Ok(IndexStatus::Complete(path.to_string()));
}

pub enum IndexStatus {
    Complete(String),
    InProgress(String),
}
