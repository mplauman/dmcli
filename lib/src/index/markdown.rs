use pulldown_cmark::{Event, HeadingLevel, Parser, Tag, TagEnd};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct ChunkPayload<'a> {
    pub content: String,
    pub path: &'a Path,
    pub headers: Vec<String>,
}

pub fn process_markdown<'a>(content: &'a str, path: &'a Path) -> Vec<ChunkPayload<'a>> {
    let options = pulldown_cmark::Options::ENABLE_TABLES;
    let parser = Parser::new_ext(content, options);
    let mut chunks = Vec::new();
    let mut header_stack: Vec<String> = Vec::new();
    let mut current_text = String::new();
    let mut in_header = false;
    let mut current_header_text = String::new();
    let mut current_header_level: usize = 0;

    for event in parser {
        match event {
            Event::Start(Tag::Heading { level, .. }) => {
                // Emit accumulated text as a chunk before starting a new heading.
                let trimmed = current_text.trim();
                if !trimmed.is_empty() {
                    let normalized = trimmed.split_whitespace().collect::<Vec<&str>>().join(" ");
                    chunks.push(ChunkPayload {
                        content: normalized,
                        path,
                        headers: header_stack.clone(),
                    });
                    current_text.clear();
                }
                in_header = true;
                current_header_level = match level {
                    HeadingLevel::H1 => 1,
                    HeadingLevel::H2 => 2,
                    HeadingLevel::H3 => 3,
                    HeadingLevel::H4 => 4,
                    HeadingLevel::H5 => 5,
                    HeadingLevel::H6 => 6,
                };
                current_header_text.clear();
            }
            Event::End(TagEnd::Heading(_)) => {
                in_header = false;
                let target_depth = current_header_level.saturating_sub(1);
                while header_stack.len() > target_depth {
                    header_stack.pop();
                }
                header_stack.push(current_header_text.trim().to_string());
            }
            Event::Text(text) => match in_header {
                true => current_header_text.push_str(&text),
                false => current_text.push_str(&text),
            },
            Event::Code(text) => match in_header {
                true => current_header_text.push_str(&text),
                false => current_text.push_str(&text),
            },
            Event::SoftBreak | Event::HardBreak => match in_header {
                true => current_header_text.push(' '),
                false => current_text.push('\n'),
            },
            _ => {}
        }
    }

    // Emit the final accumulated chunk.
    let trimmed = current_text.trim();
    if !trimmed.is_empty() {
        let normalized = trimmed.split_whitespace().collect::<Vec<&str>>().join(" ");
        chunks.push(ChunkPayload {
            content: normalized,
            path,
            headers: header_stack,
        });
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ignore_metadata() {
        let content = r#"---
title: Test
---
"#;
        let chunks = process_markdown(content, Path::new("test.md"));
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn test_process_markdown_simple() {
        let content = r#"---
title: Test
---
# Header 1
Some text.

## Header 2
More text.
"#;
        let chunks = process_markdown(content, Path::new("test.md"));
        assert_eq!(chunks.len(), 2);

        assert_eq!(chunks[0].headers, vec!["Header 1"]);
        assert_eq!(chunks[0].content, "Some text.");

        assert_eq!(chunks[1].headers, vec!["Header 1", "Header 2"]);
        assert_eq!(chunks[1].content, "More text.");
    }

    #[test]
    fn test_process_markdown_complex_hierarchy() {
        let content = r#"
# H1
Text 1

### H3
Text 3

## H2
Text 2
"#;
        let chunks = process_markdown(content, Path::new("test.md"));
        assert_eq!(chunks.len(), 3);

        assert_eq!(chunks[0].headers, vec!["H1"]);
        assert_eq!(chunks[0].content, "Text 1");

        assert_eq!(chunks[1].headers, vec!["H1", "H3"]);
        assert_eq!(chunks[1].content, "Text 3");

        assert_eq!(chunks[2].headers, vec!["H1", "H2"]);
        assert_eq!(chunks[2].content, "Text 2");
    }

    #[test]
    fn test_process_markdown_links_images() {
        let content = r#"
# Links
Here is a [link](https://example.com) and an ![image](img.png).
"#;
        let chunks = process_markdown(content, Path::new("test.md"));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "Here is a link and an image.");
    }

    #[test]
    fn test_hashes_map() {
        let one = process_markdown("hello, world!", Path::new("one.md"));
        let two = process_markdown("hello, world!", Path::new("two.md"));

        for (a, b) in one.into_iter().zip(two.into_iter()) {
            assert_eq!(a.content, b.content);
            assert_eq!(a.headers, b.headers);

            assert_eq!(a.path, Path::new("one.md"));
            assert_eq!(b.path, Path::new("two.md"));
        }
    }
}
