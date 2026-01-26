use std::fs::read_to_string;
use text_splitter::{ChunkConfig, MarkdownSplitter};
use tokenizers::Tokenizer;
use walkdir::{DirEntry, WalkDir};

use crate::error::Error;
use crate::result::Result;

pub fn index<const CHUNK_SIZE: usize>(path: &str, sync: bool) -> Result<IndexStatus> {
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

    let tokenizer = Tokenizer::from_pretrained("bert-base-uncased", None)
        .map_err(|e| Error::Index(format!("Failed to initialize tokenizer: {e:?}")))?;
    let chunk_config = ChunkConfig::new(CHUNK_SIZE)
        .with_sizer(&tokenizer)
        .with_overlap(32)
        .expect("Overlap is a sane value");
    let splitter = MarkdownSplitter::new(chunk_config);

    for entry in walker {
        println!("Processing {entry:?}");

        let contents = read_to_string(&entry)?;
        let chunks: Vec<&str> = splitter.chunks(&contents).collect();
        println!(" => {} chunks", chunks.len());

        let encodings = tokenizer
            .encode_batch_fast(chunks, false)
            .map_err(|e| Error::Index(format!("Failed to encode batch from {entry:?}: {e:?}")))?
            .into_iter()
            .map(|e| e.get_ids().len())
            .collect::<Vec<_>>();

        for encoding in encodings.into_iter().filter(|l| *l > 1000) {
            println!(" => {encoding:?}");
        }
    }

    return Ok(IndexStatus::Complete(path.to_string()));
}

pub enum IndexStatus {
    Complete(String),
    InProgress(String),
}
