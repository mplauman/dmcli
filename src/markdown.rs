use pulldown_cmark::{Event, Parser, Tag, TagEnd};
use textwrap::{Options, WordSeparator, WordSplitter, wrap};

#[derive(Debug, Clone)]
pub struct MarkdownRenderer {
    width: usize,
    options: Options<'static>,
}

impl MarkdownRenderer {
    pub fn new(width: usize) -> Self {
        let options = Options::new(width)
            .word_separator(WordSeparator::AsciiSpace)
            .word_splitter(WordSplitter::HyphenSplitter);

        Self { width, options }
    }

    pub fn with_width(&mut self, width: usize) -> &mut Self {
        self.width = width;
        self.options = Options::new(width)
            .word_separator(WordSeparator::AsciiSpace)
            .word_splitter(WordSplitter::HyphenSplitter);
        self
    }

    pub fn render(&self, markdown: &str) -> String {
        let parser = Parser::new(markdown);
        let mut output = String::new();
        let mut stack = Vec::new();
        let mut current_line = String::new();
        let mut in_code_block = false;
        let mut code_block_content = String::new();
        let mut list_depth: usize = 0;
        let mut ordered_list_counters = Vec::new();

        for event in parser {
            match event {
                Event::Start(tag) => {
                    match tag {
                        Tag::Paragraph => {
                            if !current_line.is_empty() {
                                self.flush_current_line(&mut output, &current_line);
                                current_line.clear();
                            }
                        }
                        Tag::Heading { level, .. } => {
                            if !current_line.is_empty() {
                                self.flush_current_line(&mut output, &current_line);
                                current_line.clear();
                            }
                            // Add prefix based on heading level
                            let prefix = "#".repeat(level as usize);
                            current_line.push_str(&format!("{prefix} "));
                        }
                        Tag::CodeBlock(_) => {
                            if !current_line.is_empty() {
                                self.flush_current_line(&mut output, &current_line);
                                current_line.clear();
                            }
                            in_code_block = true;
                            code_block_content.clear();
                        }
                        Tag::List(start_number) => {
                            if !current_line.is_empty() {
                                self.flush_current_line(&mut output, &current_line);
                                current_line.clear();
                            }
                            list_depth += 1;
                            if let Some(start) = start_number {
                                ordered_list_counters.push(start);
                            }
                        }
                        Tag::Item => {
                            if !current_line.is_empty() {
                                self.flush_current_line(&mut output, &current_line);
                                current_line.clear();
                            }
                            let indent = "  ".repeat(list_depth.saturating_sub(1));
                            if !ordered_list_counters.is_empty() {
                                let counter = ordered_list_counters.last_mut().unwrap();
                                current_line.push_str(&format!("{indent}{counter}. "));
                                *counter += 1;
                            } else {
                                current_line.push_str(&format!("{indent}• "));
                            }
                        }
                        Tag::Emphasis => current_line.push('*'),
                        Tag::Strong => current_line.push_str("**"),
                        Tag::BlockQuote(_) => {
                            if !current_line.is_empty() {
                                self.flush_current_line(&mut output, &current_line);
                                current_line.clear();
                            }
                            current_line.push_str("> ");
                        }
                        _ => {}
                    }
                    stack.push(tag);
                }
                Event::End(tag_end) => {
                    match tag_end {
                        TagEnd::Paragraph => {
                            if !current_line.is_empty() {
                                self.flush_current_line(&mut output, &current_line);
                                current_line.clear();
                            }
                            output.push('\n');
                        }
                        TagEnd::Heading { .. } => {
                            if !current_line.is_empty() {
                                self.flush_current_line(&mut output, &current_line);
                                current_line.clear();
                            }
                            output.push('\n');
                        }
                        TagEnd::CodeBlock => {
                            in_code_block = false;
                            // For code blocks, preserve formatting and don't wrap
                            if !code_block_content.is_empty() {
                                // Add indentation to each line of code block
                                for line in code_block_content.lines() {
                                    output.push_str("    ");
                                    output.push_str(line);
                                    output.push('\n');
                                }
                                output.push('\n');
                                code_block_content.clear();
                            }
                        }
                        TagEnd::List(_) => {
                            list_depth = list_depth.saturating_sub(1);
                            if !ordered_list_counters.is_empty() {
                                ordered_list_counters.pop();
                            }
                            if list_depth == 0 {
                                output.push('\n');
                            }
                        }
                        TagEnd::Item => {
                            if !current_line.is_empty() {
                                self.flush_current_line(&mut output, &current_line);
                                current_line.clear();
                            }
                        }
                        TagEnd::Emphasis => current_line.push('*'),
                        TagEnd::Strong => current_line.push_str("**"),
                        TagEnd::BlockQuote(_) => {
                            if !current_line.is_empty() {
                                self.flush_current_line(&mut output, &current_line);
                                current_line.clear();
                            }
                            output.push('\n');
                        }
                        _ => {}
                    }
                    stack.pop();
                }
                Event::Text(text) => {
                    if in_code_block {
                        code_block_content.push_str(&text);
                    } else {
                        current_line.push_str(&text);
                    }
                }
                Event::Code(code) => {
                    current_line.push('`');
                    current_line.push_str(&code);
                    current_line.push('`');
                }
                Event::SoftBreak => {
                    current_line.push(' ');
                }
                Event::HardBreak => {
                    if !current_line.is_empty() {
                        self.flush_current_line(&mut output, &current_line);
                        current_line.clear();
                    }
                }
                Event::Rule => {
                    if !current_line.is_empty() {
                        self.flush_current_line(&mut output, &current_line);
                        current_line.clear();
                    }
                    let rule = "─".repeat(self.width.min(80));
                    output.push_str(&rule);
                    output.push('\n');
                    output.push('\n');
                }
                _ => {}
            }
        }

        // Flush any remaining content
        if !current_line.is_empty() {
            self.flush_current_line(&mut output, &current_line);
        }

        // Clean up extra newlines at the end
        output.trim_end().to_string()
    }

    fn flush_current_line(&self, output: &mut String, line: &str) {
        if line.trim().is_empty() {
            return;
        }

        // Check if this is a special line that shouldn't be wrapped
        let trimmed = line.trim();
        if trimmed.starts_with("    ") || // Code block
           trimmed.starts_with("```") ||  // Code fence
           trimmed.starts_with("---") ||  // Horizontal rule
           trimmed.starts_with("===")
        // Horizontal rule
        {
            output.push_str(line);
            output.push('\n');
            return;
        }

        // Handle list items and block quotes specially
        if let Some(prefix_end) = self.find_prefix_end(line) {
            let prefix = &line[..prefix_end];
            let content = &line[prefix_end..];

            if content.trim().is_empty() {
                output.push_str(line);
                output.push('\n');
                return;
            }

            let wrapped_content = wrap(content.trim(), &self.options);
            if wrapped_content.is_empty() {
                output.push_str(line);
                output.push('\n');
                return;
            }

            // First line with prefix
            output.push_str(prefix);
            output.push_str(&wrapped_content[0]);
            output.push('\n');

            // Subsequent lines with hanging indent
            let hanging_indent = " ".repeat(prefix.len());
            for wrapped_line in wrapped_content.iter().skip(1) {
                output.push_str(&hanging_indent);
                output.push_str(wrapped_line);
                output.push('\n');
            }
        } else {
            // Regular line - wrap normally
            let wrapped = wrap(line, &self.options);
            for wrapped_line in wrapped {
                output.push_str(&wrapped_line);
                output.push('\n');
            }
        }
    }

    fn find_prefix_end(&self, line: &str) -> Option<usize> {
        let trimmed = line.trim_start();
        let indent_len = line.len() - trimmed.len();

        // Check for list items
        if let Some(pos) = trimmed.find("• ") {
            return Some(indent_len + pos + "• ".len());
        }

        // Check for numbered list items
        if let Some(pos) = trimmed.find(". ") {
            let before_dot = &trimmed[..pos];
            if before_dot.chars().all(|c| c.is_ascii_digit()) {
                return Some(indent_len + pos + ". ".len());
            }
        }

        // Check for block quotes
        if trimmed.starts_with("> ") {
            return Some(indent_len + "> ".len());
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_paragraph() {
        let renderer = MarkdownRenderer::new(20);
        let input =
            "This is a long paragraph that should be wrapped to fit within the specified width.";
        let output = renderer.render(input);

        assert!(output.lines().all(|line| line.len() <= 20));
        assert!(output.contains("This is a long"));
    }

    #[test]
    fn test_heading() {
        let renderer = MarkdownRenderer::new(50);
        let input = "# Main Heading\n\n## Sub Heading";
        let output = renderer.render(input);

        assert!(output.contains("# Main Heading"));
        assert!(output.contains("## Sub Heading"));
    }

    #[test]
    fn test_code_block() {
        let renderer = MarkdownRenderer::new(20);
        let input = "```\nlet x = 42;\nprintln!(\"Hello\");\n```";
        let output = renderer.render(input);

        assert!(output.contains("    let x = 42;"));
        assert!(output.contains("    println!(\"Hello\");"));
    }

    #[test]
    fn test_list() {
        let renderer = MarkdownRenderer::new(30);
        let input = "- First item\n- Second item with longer text\n- Third item";
        let output = renderer.render(input);

        assert!(output.contains("• First item"));
        assert!(output.contains("• Second item"));
        assert!(output.contains("• Third item"));
    }

    #[test]
    fn test_ordered_list() {
        let renderer = MarkdownRenderer::new(30);
        let input = "1. First item\n2. Second item\n3. Third item";
        let output = renderer.render(input);

        assert!(output.contains("1. First item"));
        assert!(output.contains("2. Second item"));
        assert!(output.contains("3. Third item"));
    }

    #[test]
    fn test_width_change() {
        let mut renderer = MarkdownRenderer::new(10);
        let input = "This is a test of changing width";

        let output1 = renderer.render(input);
        assert!(output1.lines().all(|line| line.len() <= 10));

        renderer.with_width(30);
        let output2 = renderer.render(input);
        assert!(output2.lines().all(|line| line.len() <= 30));
        assert!(output2.lines().count() < output1.lines().count());
    }
}
