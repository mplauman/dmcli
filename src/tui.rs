use crate::errors::Error;
use crate::events::AppEvent;
use crate::markdown::MarkdownRenderer;
use config::Config;
use crossterm::{
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
};
use std::{
    collections::VecDeque,
    io::{self, Stdout},
};

#[derive(Clone, Debug)]
pub struct ConversationMessage {
    pub content: String,
    pub message_type: MessageType,
    pub lines: Option<Vec<String>>,
}

#[derive(Clone, Debug)]
pub enum MessageType {
    User,
    Assistant,
    System,
    Thinking,
    Error,
}

pub struct Tui {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    current_line: String,
    cursor_position: usize,
    conversation: VecDeque<ConversationMessage>,
    input_height: u16,
    scroll_offset: u16,
    terminal_width: u16,
    terminal_height: u16,
    markdown_renderer: MarkdownRenderer,
    search_mode: bool,
    search_query: String,
    search_results: Vec<(usize, String)>,
    search_index: usize,
}

impl Tui {
    pub fn new(
        _config: &Config,
        _event_sender: async_channel::Sender<AppEvent>,
    ) -> Result<Self, Error> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;

        let size = terminal.size()?;

        let mut tui = Self {
            terminal,
            current_line: String::new(),
            cursor_position: 0,
            conversation: VecDeque::new(),
            input_height: 3,
            scroll_offset: 0,
            terminal_width: size.width,
            terminal_height: size.height,
            markdown_renderer: MarkdownRenderer::new(size.width.saturating_sub(4) as usize),
            search_mode: false,
            search_query: String::new(),
            search_results: Vec::new(),
            search_index: 0,
        };

        // Add welcome message
        tui.add_message(
            "Welcome to dmcli! Type your message and press Enter to send. Send 'roll 2d6' to roll a dice or 'exit' to quit.".to_string(),
            MessageType::System,
        );

        Ok(tui)
    }

    pub fn render(&mut self) -> Result<(), Error> {
        self.update_input_height();

        let size = ratatui::layout::Rect {
            x: 0,
            y: 0,
            width: self.terminal_width,
            height: self.terminal_height,
        };

        // Create layout with conversation on top and input on bottom
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(self.input_height)])
            .split(size);

        let search_status = self.get_search_status();
        let paragraph = Self::render_paragraph(
            &mut self.conversation,
            &mut self.markdown_renderer,
            &mut self.scroll_offset,
            chunks[0],
            search_status,
        );

        self.terminal.draw(|f| {
            Self::render_ui_static(
                f,
                self.input_height,
                &self.current_line,
                self.cursor_position,
                paragraph,
                chunks[0],
                self.search_mode,
            );
        })?;

        Ok(())
    }

    pub fn add_message(&mut self, content: String, message_type: MessageType) {
        self.conversation.push_back(ConversationMessage {
            content,
            message_type,
            lines: None,
        });

        // Keep conversation history reasonable (last 100 messages)
        if self.conversation.len() > 100 {
            self.conversation.pop_front();
        }

        // Auto-scroll to bottom when new messages are added
        self.scroll_offset = 0;
    }

    pub fn input_updated(&mut self, current_line: String, cursor_position: usize) {
        self.current_line = current_line;
        self.cursor_position = cursor_position;
        self.update_input_height();
    }

    pub fn resized(&mut self, width: u16, height: u16) {
        log::debug!("Window resized: {width}x{height}");
        self.terminal_width = width;
        self.terminal_height = height;
        self.markdown_renderer
            .with_width(width.saturating_sub(4) as usize);
        self.update_input_height();

        for message in self.conversation.iter_mut() {
            message.lines = None;
        }
    }

    pub fn handle_scroll_back(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_add(10_u16);
    }

    pub fn handle_scroll_forward(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_sub(10_u16);
    }

    /// Enter search mode and start searching the conversation history
    pub fn start_search(&mut self) {
        self.search_mode = true;
        self.search_query.clear();
        self.search_results.clear();
        self.search_index = 0;
    }

    /// Exit search mode and return to normal conversation view
    pub fn exit_search(&mut self) {
        self.search_mode = false;
        self.search_query.clear();
        self.search_results.clear();
        self.search_index = 0;
    }

    /// Update search query and refresh results
    pub fn update_search_query(&mut self, query: String) {
        self.search_query = query;
        self.search_results.clear();
        self.search_index = 0;
        
        if !self.search_query.is_empty() {
            self.perform_search();
        }
    }

    /// Perform the actual search through conversation history
    fn perform_search(&mut self) {
        let query = self.search_query.to_lowercase();
        
        for (index, message) in self.conversation.iter().enumerate() {
            if message.content.to_lowercase().contains(&query) {
                // Create a snippet with context around the match
                let snippet = self.create_search_snippet(&message.content, &query);
                self.search_results.push((index, snippet));
            }
        }
    }

    /// Create a snippet showing the search match with context
    fn create_search_snippet(&self, content: &str, query: &str) -> String {
        let content_lower = content.to_lowercase();
        let query_lower = query.to_lowercase();
        
        if let Some(pos) = content_lower.find(&query_lower) {
            let start = pos.saturating_sub(30);
            let end = (pos + query.len() + 30).min(content.len());
            let snippet = &content[start..end];
            
            if start > 0 {
                format!("...{}", snippet)
            } else {
                snippet.to_string()
            }
        } else {
            content.chars().take(60).collect()
        }
    }

    /// Navigate to the next search result
    pub fn next_search_result(&mut self) {
        if !self.search_results.is_empty() {
            self.search_index = (self.search_index + 1) % self.search_results.len();
            self.jump_to_search_result();
        }
    }

    /// Navigate to the previous search result
    pub fn prev_search_result(&mut self) {
        if !self.search_results.is_empty() {
            self.search_index = if self.search_index > 0 {
                self.search_index - 1
            } else {
                self.search_results.len() - 1
            };
            self.jump_to_search_result();
        }
    }

    /// Jump to the current search result by adjusting scroll offset
    fn jump_to_search_result(&mut self) {
        if let Some((message_index, _)) = self.search_results.get(self.search_index) {
            // Calculate scroll offset to bring the message into view
            let total_messages = self.conversation.len();
            let scroll_position = total_messages.saturating_sub(message_index + 1);
            self.scroll_offset = scroll_position as u16;
        }
    }

    /// Get the current search status for display
    pub fn get_search_status(&self) -> Option<String> {
        if self.search_mode {
            if self.search_results.is_empty() {
                if self.search_query.is_empty() {
                    Some("Search: ".to_string())
                } else {
                    Some(format!("Search: {} (no results)", self.search_query))
                }
            } else {
                Some(format!(
                    "Search: {} ({}/{} results)",
                    self.search_query,
                    self.search_index + 1,
                    self.search_results.len()
                ))
            }
        } else {
            None
        }
    }

    fn update_input_height(&mut self) {
        let available_width = self.terminal_width.saturating_sub(4); // Account for borders

        let lines = if self.current_line.is_empty() {
            1
        } else {
            self.current_line
                .lines()
                .map(|line| {
                    if line.len() as u16 <= available_width {
                        1
                    } else {
                        (line.len() as u16 + available_width - 1).div_ceil(available_width)
                    }
                })
                .sum::<u16>()
                .max(1)
        };

        self.input_height = (lines + 2).min(10); // Cap at 10 lines maximum
    }

    fn render_ui_static(
        f: &mut Frame,
        input_height: u16,
        current_line: &str,
        cursor_position: usize,
        paragraph: Paragraph<'_>,
        paragraph_rect: ratatui::layout::Rect,
        search_mode: bool,
    ) {
        let size = f.area();

        // Create layout with conversation on top and input on bottom
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(input_height)])
            .split(size);

        // Render conversation window
        f.render_widget(paragraph, paragraph_rect);

        // Render input window
        Self::render_input_static(f, chunks[1], current_line, cursor_position, search_mode);
    }

    fn render_paragraph(
        conversation: &mut VecDeque<ConversationMessage>,
        markdown_renderer: &mut MarkdownRenderer,
        in_scroll_offset: &mut u16,
        area: ratatui::layout::Rect,
        search_status: Option<String>,
    ) -> Paragraph<'static> {
        let mut lines: VecDeque<Line<'static>> = VecDeque::with_capacity(area.height as usize - 2);
        let mut scroll_offset = *in_scroll_offset;

        let rendered_lines = conversation
            .iter_mut()
            .rev()
            .flat_map(|msg| {
                let style = match msg.message_type {
                    MessageType::User => Style::default().fg(Color::Cyan),
                    MessageType::Assistant => Style::default().fg(Color::Green),
                    MessageType::System => Style::default().fg(Color::Yellow),
                    MessageType::Thinking => Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::ITALIC),
                    MessageType::Error => Style::default().fg(Color::Red),
                };

                let rendered_content = if let Some(cached) = msg.lines.as_ref() {
                    cached.clone()
                } else {
                    let rendered_content = markdown_renderer
                        .render(&msg.content)
                        .lines()
                        .map(str::to_owned)
                        .collect::<Vec<_>>();

                    msg.lines = Some(rendered_content.clone());
                    rendered_content
                };

                // Split the rendered content into lines and apply styling
                rendered_content
                    .into_iter()
                    .map(|line| Line::from(vec![Span::styled(line, style)]))
                    .chain(std::iter::once(Line::from("")))
                    .rev()
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        for line in rendered_lines {
            if lines.len() == area.height as usize - 2 {
                // Window is filled up. If there's still scroll offset left then drop the oldest line,
                // otherwise break.
                if scroll_offset == 0 {
                    break;
                }

                scroll_offset -= 1;
                lines.pop_back();
            }

            lines.push_front(line);
        }

        // Fixes up any over-scrolling
        *in_scroll_offset -= scroll_offset;

        let text = Text::from(lines.into_iter().collect::<Vec<_>>());

        let title = if let Some(search_status) = search_status {
            format!("Conversation (PgUp/PgDn: scroll, Ctrl+F: search) - {}", search_status)
        } else {
            "Conversation (PgUp/PgDn: scroll, Ctrl+F: search)".to_string()
        };
        let conversation_block = Block::default().borders(Borders::ALL).title(title);

        Paragraph::new(text).block(conversation_block)
    }

    fn render_input_static(
        f: &mut Frame,
        area: ratatui::layout::Rect,
        current_line: &str,
        cursor_position: usize,
        search_mode: bool,
    ) {
        let title = if search_mode {
            "Search (Esc: exit | Ctrl+G: next | Ctrl+Shift+G: prev)"
        } else {
            "Input (Enter: send | 'exit': quit | Ctrl+F: search)"
        };
        let input_block = Block::default()
            .borders(Borders::ALL)
            .title(title)
            .style(if search_mode {
                Style::default().fg(Color::Magenta)
            } else {
                Style::default().fg(Color::Green)
            });

        let input_area = input_block.inner(area);

        let input_text = if current_line.is_empty() {
            if search_mode {
                Text::from("Type search query...")
            } else {
                Text::from("Type your message here... Press Enter to send")
            }
        } else {
            Text::from(current_line)
        };

        let style = if current_line.is_empty() {
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC)
        } else {
            Style::default()
        };

        let paragraph = Paragraph::new(input_text)
            .style(style)
            .wrap(Wrap { trim: false });

        f.render_widget(input_block, area);
        f.render_widget(paragraph, input_area);

        // Calculate cursor position for display
        if !current_line.is_empty() {
            let available_width = input_area.width as usize;
            let lines_before_cursor: usize = current_line[..cursor_position]
                .lines()
                .enumerate()
                .map(|(i, line)| {
                    if i == 0 {
                        line.len() / available_width
                    } else {
                        (line.len() + available_width - 1).div_ceil(available_width)
                    }
                })
                .sum();

            let current_line_pos = current_line[..cursor_position]
                .lines()
                .last()
                .unwrap_or("")
                .len()
                % available_width;

            f.set_cursor_position((
                input_area.x + current_line_pos as u16,
                input_area.y + lines_before_cursor as u16,
            ));
        } else {
            f.set_cursor_position((input_area.x, input_area.y));
        }
    }
}

impl Drop for Tui {
    fn drop(&mut self) {
        // Restore terminal
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use config::Config;
    use async_channel;

    fn create_test_tui() -> Tui {
        let _config = Config::default();
        let (_tx, _rx) = async_channel::unbounded::<AppEvent>();
        
        // Create a mock terminal backend for testing
        let backend = CrosstermBackend::new(std::io::stdout());
        let terminal = Terminal::new(backend).unwrap();
        
        Tui {
            terminal,
            current_line: String::new(),
            cursor_position: 0,
            conversation: VecDeque::new(),
            input_height: 3,
            scroll_offset: 0,
            terminal_width: 80,
            terminal_height: 24,
            markdown_renderer: MarkdownRenderer::new(76),
            search_mode: false,
            search_query: String::new(),
            search_results: Vec::new(),
            search_index: 0,
        }
    }

    #[test]
    fn test_search_mode_activation() {
        let mut tui = create_test_tui();
        
        // Initially not in search mode
        assert!(!tui.search_mode);
        assert!(tui.get_search_status().is_none());
        
        // Start search mode
        tui.start_search();
        assert!(tui.search_mode);
        assert!(tui.get_search_status().is_some());
        
        // Exit search mode
        tui.exit_search();
        assert!(!tui.search_mode);
        assert!(tui.get_search_status().is_none());
    }

    #[test]
    fn test_conversation_search() {
        let mut tui = create_test_tui();
        
        // Add some test messages
        tui.add_message("Hello world".to_string(), MessageType::User);
        tui.add_message("How are you?".to_string(), MessageType::Assistant);
        tui.add_message("I'm doing well, thanks for asking".to_string(), MessageType::User);
        
        // Start search and search for "hello"
        tui.start_search();
        tui.update_search_query("hello".to_string());
        
        // Should find one result
        assert_eq!(tui.search_results.len(), 1);
        assert_eq!(tui.search_index, 0);
        
        // Search for "are"
        tui.update_search_query("are".to_string());
        assert_eq!(tui.search_results.len(), 1);
        
        // Search for "thanks"
        tui.update_search_query("thanks".to_string());
        assert_eq!(tui.search_results.len(), 1);
        
        // Search for non-existent text
        tui.update_search_query("nonexistent".to_string());
        assert_eq!(tui.search_results.len(), 0);
    }

    #[test]
    fn test_search_navigation() {
        let mut tui = create_test_tui();
        
        // Add messages with repeated search terms
        tui.add_message("The quick brown fox".to_string(), MessageType::User);
        tui.add_message("The lazy dog".to_string(), MessageType::Assistant);
        tui.add_message("The end of the story".to_string(), MessageType::User);
        
        // Search for "the"
        tui.start_search();
        tui.update_search_query("the".to_string());
        
        // Should find multiple results
        assert_eq!(tui.search_results.len(), 3);
        assert_eq!(tui.search_index, 0);
        
        // Navigate to next result
        tui.next_search_result();
        assert_eq!(tui.search_index, 1);
        
        // Navigate to next result
        tui.next_search_result();
        assert_eq!(tui.search_index, 2);
        
        // Navigate to next result (should wrap to 0)
        tui.next_search_result();
        assert_eq!(tui.search_index, 0);
        
        // Navigate to previous result
        tui.prev_search_result();
        assert_eq!(tui.search_index, 2);
    }

    #[test]
    fn test_search_snippet_creation() {
        let tui = create_test_tui();
        
        // Test snippet creation for short content
        let snippet = tui.create_search_snippet("Hello world", "world");
        assert_eq!(snippet, "Hello world");
        
        // Test snippet creation for long content
        let long_content = "This is a very long message that should be truncated to show only the relevant part around the search term for better user experience when searching through conversation history.";
        let snippet = tui.create_search_snippet(long_content, "relevant");
        assert!(snippet.contains("relevant"));
        assert!(snippet.len() < long_content.len());
    }

    #[test]
    fn test_search_status_display() {
        let mut tui = create_test_tui();
        
        // Initially no search status
        assert!(tui.get_search_status().is_none());
        
        // Start search mode
        tui.start_search();
        let status = tui.get_search_status().unwrap();
        assert_eq!(status, "Search: ");
        
        // Update search query
        tui.update_search_query("test".to_string());
        let status = tui.get_search_status().unwrap();
        assert_eq!(status, "Search: test (no results)");
        
        // Add a message and search
        tui.add_message("This is a test message".to_string(), MessageType::User);
        tui.update_search_query("test".to_string());
        let status = tui.get_search_status().unwrap();
        assert_eq!(status, "Search: test (1/1 results)");
    }
}
