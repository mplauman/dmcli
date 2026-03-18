use std::fmt;
use std::path::PathBuf;

/// The source of a search result.
#[derive(Debug, Clone, PartialEq)]
pub enum Source {
    /// A file on disk, with a section identifier (e.g. a heading or anchor).
    File { path: PathBuf, section: String },
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Source::File { path, section } => {
                write!(f, "{}#{}", path.display(), section)
            }
        }
    }
}

/// A single result returned from a similarity search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// How closely this result matches the search term. Higher is closer.
    pub score: f32,
    /// The text content of the matching chunk.
    pub text: String,
    /// The origin of the text, if known.
    pub source: Option<Source>,
}

impl fmt::Display for SearchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(source) = &self.source {
            writeln!(f, "Source:  {source}")?;
        }
        writeln!(f, "Score:   {:.4}", self.score)?;
        write!(f, "{}", self.text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn file_source() -> Source {
        Source::File {
            path: PathBuf::from("docs/rules.md"),
            section: "combat".to_string(),
        }
    }

    #[test]
    fn source_display() {
        let s = file_source();
        assert_eq!(s.to_string(), "docs/rules.md#combat");
    }

    #[test]
    fn search_result_display_with_source() {
        let result = SearchResult {
            score: 0.9123,
            text: "Roll 2d6 for initiative.".to_string(),
            source: Some(file_source()),
        };

        let output = result.to_string();
        assert!(output.contains("docs/rules.md#combat"));
        assert!(output.contains("0.9123"));
        assert!(output.contains("Roll 2d6 for initiative."));
    }

    #[test]
    fn search_result_display_without_source() {
        let result = SearchResult {
            score: 0.5,
            text: "Some orphaned text.".to_string(),
            source: None,
        };

        let output = result.to_string();
        assert!(!output.contains("Source:"));
        assert!(output.contains("0.5000"));
        assert!(output.contains("Some orphaned text."));
    }
}
