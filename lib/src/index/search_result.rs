use std::fmt;
use std::fmt::Write as FmtWrite;
use std::path::PathBuf;

use crate::result::Result;

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

/// A ranked collection of [`SearchResult`]s with three output modes:
///
/// - [`fmt::Display`]: Markdown document, intended for human consumption at a
///   terminal or in an editor. Results are numbered with `## [N]` headings and
///   separated by horizontal rules. The relevance score is omitted.
///
/// - [`SearchResults::to_xml`]: XML-style document blocks suitable for
///   injection into an LLM prompt as RAG context. Each chunk is wrapped in a
///   `<document>` element with an `index` attribute for citation.
///
/// - [`SearchResults::to_json`]: JSON array with `rank`, `score`, `source`,
///   and `text` fields for structured agent-skill responses.
pub struct SearchResults(pub Vec<SearchResult>);

impl From<Vec<SearchResult>> for SearchResults {
    fn from(v: Vec<SearchResult>) -> Self {
        Self(v)
    }
}

impl fmt::Display for SearchResults {
    /// Renders results as a Markdown document for human-readable output.
    ///
    /// Each result is introduced with a `## [N] <source>` heading and
    /// separated from the next by a horizontal rule (`---`). An empty result
    /// set renders as `*No results found.*`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return write!(f, "*No results found.*");
        }

        for (i, result) in self.0.iter().enumerate() {
            let index = i + 1;
            match &result.source {
                Some(source) => writeln!(f, "## [{index}] {source}")?,
                None => writeln!(f, "## [{index}]")?,
            }
            writeln!(f)?;
            writeln!(f, "{}", result.text.trim())?;

            if index < self.0.len() {
                writeln!(f)?;
                writeln!(f, "---")?;
                writeln!(f)?;
            }
        }

        Ok(())
    }
}

impl SearchResults {
    /// Renders results as XML-style document blocks for injection into an LLM
    /// prompt as RAG context.
    ///
    /// Each result is wrapped in a `<document index="N">` element containing
    /// an optional `<source>` tag and a `<content>` tag. The relevance score
    /// is intentionally omitted — it is an implementation detail that adds
    /// noise to the model context.
    pub fn to_xml(&self) -> String {
        let mut buf = String::new();

        for (i, result) in self.0.iter().enumerate() {
            let index = i + 1;
            writeln!(buf, "<document index=\"{index}\">").expect("write to String is infallible");
            if let Some(source) = &result.source {
                writeln!(buf, "<source>{source}</source>").expect("write to String is infallible");
            }
            writeln!(buf, "<content>").expect("write to String is infallible");
            writeln!(buf, "{}", result.text.trim()).expect("write to String is infallible");
            writeln!(buf, "</content>").expect("write to String is infallible");
            write!(buf, "</document>").expect("write to String is infallible");

            if index < self.0.len() {
                writeln!(buf).expect("write to String is infallible");
                writeln!(buf).expect("write to String is infallible");
            }
        }

        buf
    }

    /// Serializes the results to a pretty-printed JSON string for use in
    /// agent-skill responses.
    ///
    /// Each object in the array contains:
    ///
    /// | Field    | Type            | Description                                     |
    /// |----------|-----------------|-------------------------------------------------|
    /// | `rank`   | integer         | 1-based position; lower rank = higher relevance |
    /// | `score`  | number          | Raw similarity score from the vector store      |
    /// | `source` | string \| null  | Origin of the chunk; `null` when unknown        |
    /// | `text`   | string          | The chunk text                                  |
    pub fn to_json(&self) -> Result<String> {
        let values: Vec<serde_json::Value> = self
            .0
            .iter()
            .enumerate()
            .map(|(i, r)| {
                serde_json::json!({
                    "rank": i + 1,
                    "score": r.score,
                    "source": r.source.as_ref().map(|s| s.to_string()),
                    "text": r.text,
                })
            })
            .collect();

        serde_json::to_string_pretty(&values).map_err(crate::error::Error::from)
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

    fn make_result(score: f32, text: &str, source: Option<Source>) -> SearchResult {
        SearchResult {
            score,
            text: text.to_string(),
            source,
        }
    }

    // -------------------------------------------------------------------------
    // Source
    // -------------------------------------------------------------------------

    #[test]
    fn source_display() {
        assert_eq!(file_source().to_string(), "docs/rules.md#combat");
    }

    // -------------------------------------------------------------------------
    // SearchResult — single-item debug display
    // -------------------------------------------------------------------------

    #[test]
    fn search_result_display_with_source() {
        let result = make_result(0.9123, "Roll 2d6 for initiative.", Some(file_source()));
        let output = result.to_string();
        assert!(output.contains("docs/rules.md#combat"));
        assert!(output.contains("0.9123"));
        assert!(output.contains("Roll 2d6 for initiative."));
    }

    #[test]
    fn search_result_display_without_source() {
        let result = make_result(0.5, "Some orphaned text.", None);
        let output = result.to_string();
        assert!(!output.contains("Source:"));
        assert!(output.contains("0.5000"));
        assert!(output.contains("Some orphaned text."));
    }

    // -------------------------------------------------------------------------
    // SearchResults::Display — Markdown output
    // -------------------------------------------------------------------------

    #[test]
    fn markdown_single_with_source() {
        let results = SearchResults::from(vec![make_result(
            0.9,
            "Roll 2d6 for initiative.",
            Some(file_source()),
        )]);

        let output = results.to_string();
        assert!(output.contains("## [1] docs/rules.md#combat"));
        assert!(output.contains("Roll 2d6 for initiative."));
    }

    #[test]
    fn markdown_single_no_source() {
        let results = SearchResults::from(vec![make_result(0.5, "Orphaned text.", None)]);
        let output = results.to_string();
        assert!(output.contains("## [1]"));
        assert!(output.contains("Orphaned text."));
    }

    #[test]
    fn markdown_multiple_are_numbered() {
        let results = SearchResults::from(vec![
            make_result(0.9, "First chunk.", Some(file_source())),
            make_result(0.7, "Second chunk.", None),
        ]);

        let output = results.to_string();
        assert!(output.contains("## [1]"));
        assert!(output.contains("## [2]"));
        assert!(output.contains("First chunk."));
        assert!(output.contains("Second chunk."));
    }

    #[test]
    fn markdown_multiple_have_separator() {
        let results = SearchResults::from(vec![
            make_result(0.9, "First.", Some(file_source())),
            make_result(0.7, "Second.", None),
        ]);

        assert!(results.to_string().contains("\n---\n"));
    }

    #[test]
    fn markdown_no_trailing_separator() {
        let results = SearchResults::from(vec![
            make_result(0.9, "First.", None),
            make_result(0.7, "Second.", None),
        ]);

        let output = results.to_string();
        // The horizontal rule must not appear after the final result.
        let last_doc = output.rfind("Second.").expect("second result present");
        let after_last = &output[last_doc..];
        assert!(!after_last.contains("---"));
    }

    #[test]
    fn markdown_empty_shows_no_results_message() {
        assert_eq!(
            SearchResults::from(vec![]).to_string(),
            "*No results found.*"
        );
    }

    #[test]
    fn markdown_omits_score() {
        let results = SearchResults::from(vec![make_result(
            0.9123,
            "Roll 2d6 for initiative.",
            Some(file_source()),
        )]);

        assert!(!results.to_string().contains("0.9123"));
    }

    #[test]
    fn markdown_trims_content_whitespace() {
        let results = SearchResults::from(vec![make_result(
            0.8,
            "  trimmed text  ",
            Some(file_source()),
        )]);
        let output = results.to_string();
        assert!(output.contains("trimmed text"));
        assert!(!output.contains("  trimmed text  "));
    }

    // -------------------------------------------------------------------------
    // SearchResults::to_xml — RAG context blocks
    // -------------------------------------------------------------------------

    #[test]
    fn xml_single_with_source() {
        let results = SearchResults::from(vec![make_result(
            0.9,
            "Roll 2d6 for initiative.",
            Some(file_source()),
        )]);

        let output = results.to_xml();
        assert!(output.contains("<document index=\"1\">"));
        assert!(output.contains("<source>docs/rules.md#combat</source>"));
        assert!(output.contains("<content>"));
        assert!(output.contains("Roll 2d6 for initiative."));
        assert!(output.contains("</content>"));
        assert!(output.contains("</document>"));
    }

    #[test]
    fn xml_no_source_omits_source_tag() {
        let results = SearchResults::from(vec![make_result(0.5, "Orphaned text.", None)]);
        let output = results.to_xml();
        assert!(!output.contains("<source>"));
        assert!(output.contains("<content>"));
        assert!(output.contains("Orphaned text."));
    }

    #[test]
    fn xml_multiple_are_numbered() {
        let results = SearchResults::from(vec![
            make_result(0.9, "First chunk.", Some(file_source())),
            make_result(0.7, "Second chunk.", None),
        ]);

        let output = results.to_xml();
        assert!(output.contains("index=\"1\""));
        assert!(output.contains("index=\"2\""));
        assert!(output.contains("First chunk."));
        assert!(output.contains("Second chunk."));
    }

    #[test]
    fn xml_empty_is_empty_string() {
        assert_eq!(SearchResults::from(vec![]).to_xml(), "");
    }

    #[test]
    fn xml_omits_score() {
        let results = SearchResults::from(vec![make_result(
            0.9123,
            "Roll 2d6 for initiative.",
            Some(file_source()),
        )]);

        assert!(!results.to_xml().contains("0.9123"));
    }

    #[test]
    fn xml_trims_content_whitespace() {
        let results = SearchResults::from(vec![make_result(
            0.8,
            "  trimmed text  ",
            Some(file_source()),
        )]);
        let output = results.to_xml();
        assert!(output.contains("trimmed text"));
        assert!(!output.contains("  trimmed text  "));
    }

    // -------------------------------------------------------------------------
    // SearchResults::to_json — agent-skill output
    // -------------------------------------------------------------------------

    #[test]
    fn to_json_contains_rank_and_score() {
        let results = SearchResults::from(vec![make_result(
            0.9123,
            "Roll 2d6 for initiative.",
            Some(file_source()),
        )]);

        let json = results.to_json().expect("to_json failed");
        let parsed: Vec<serde_json::Value> =
            serde_json::from_str(&json).expect("invalid JSON output");

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["rank"], 1);
        // f32 serialization may round; check proximity
        let score = parsed[0]["score"].as_f64().expect("score is a number");
        assert!((score - 0.9123_f64).abs() < 1e-4);
    }

    #[test]
    fn to_json_ranks_are_sequential() {
        let results = SearchResults::from(vec![
            make_result(0.9, "First.", Some(file_source())),
            make_result(0.7, "Second.", None),
            make_result(0.5, "Third.", None),
        ]);

        let json = results.to_json().expect("to_json failed");
        let parsed: Vec<serde_json::Value> =
            serde_json::from_str(&json).expect("invalid JSON output");

        assert_eq!(parsed[0]["rank"], 1);
        assert_eq!(parsed[1]["rank"], 2);
        assert_eq!(parsed[2]["rank"], 3);
    }

    #[test]
    fn to_json_source_present_when_known() {
        let results = SearchResults::from(vec![make_result(0.9, "Roll 2d6.", Some(file_source()))]);

        let json = results.to_json().expect("to_json failed");
        let parsed: Vec<serde_json::Value> =
            serde_json::from_str(&json).expect("invalid JSON output");

        assert_eq!(parsed[0]["source"], "docs/rules.md#combat");
    }

    #[test]
    fn to_json_source_null_when_unknown() {
        let results = SearchResults::from(vec![make_result(0.5, "Orphaned.", None)]);

        let json = results.to_json().expect("to_json failed");
        let parsed: Vec<serde_json::Value> =
            serde_json::from_str(&json).expect("invalid JSON output");

        assert!(parsed[0]["source"].is_null());
    }

    #[test]
    fn to_json_text_preserved() {
        let text = "Druids can wildshape into animals.";
        let results = SearchResults::from(vec![make_result(0.8, text, None)]);

        let json = results.to_json().expect("to_json failed");
        let parsed: Vec<serde_json::Value> =
            serde_json::from_str(&json).expect("invalid JSON output");

        assert_eq!(parsed[0]["text"], text);
    }

    #[test]
    fn to_json_empty_is_empty_array() {
        let results = SearchResults::from(vec![]);
        let json = results.to_json().expect("to_json failed");
        let parsed: Vec<serde_json::Value> =
            serde_json::from_str(&json).expect("invalid JSON output");
        assert!(parsed.is_empty());
    }
}
