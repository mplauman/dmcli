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

        let paragraph = Self::render_paragraph(
            &mut self.conversation,
            &mut self.markdown_renderer,
            &mut self.scroll_offset,
            chunks[0],
        );

        self.terminal.draw(|f| {
            Self::render_ui_static(
                f,
                self.input_height,
                &self.current_line,
                self.cursor_position,
                paragraph,
                chunks[0],
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
        Self::render_input_static(f, chunks[1], current_line, cursor_position);
    }

    fn render_paragraph(
        conversation: &mut VecDeque<ConversationMessage>,
        markdown_renderer: &mut MarkdownRenderer,
        in_scroll_offset: &mut u16,
        area: ratatui::layout::Rect,
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

        let title = "Conversation (PgUp/PgDn: scroll)";
        let conversation_block = Block::default().borders(Borders::ALL).title(title);

        Paragraph::new(text).block(conversation_block)
    }

    fn render_input_static(
        f: &mut Frame,
        area: ratatui::layout::Rect,
        current_line: &str,
        cursor_position: usize,
    ) {
        let title = "Input (Enter: send | 'exit': quit)";
        let input_block = Block::default()
            .borders(Borders::ALL)
            .title(title)
            .style(Style::default().fg(Color::Green));

        let input_area = input_block.inner(area);

        let input_text = if current_line.is_empty() {
            Text::from("Type your message here... Press Enter to send")
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
