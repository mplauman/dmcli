use crate::conversation::{Conversation, Id, Message};
use crate::embeddings::EmbeddingGenerator;
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
    collections::HashMap,
    collections::VecDeque,
    io::{self, Stdout},
};

pub struct Tui {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    formatted: HashMap<Id, Vec<String>>,
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

        let tui = Self {
            terminal,
            formatted: HashMap::new(),
            scroll_offset: 0,
            terminal_width: size.width,
            terminal_height: size.height,
            markdown_renderer: MarkdownRenderer::new(size.width.saturating_sub(4) as usize),
        };

        Ok(tui)
    }

    pub fn render(
        &mut self,
        conversation: &Conversation<impl EmbeddingGenerator>,
        input: &str,
        cursor: usize,
    ) -> Result<(), Error> {
        let input_height = self.calculate_input_height(input);

        let size = ratatui::layout::Rect {
            x: 0,
            y: 0,
            width: self.terminal_width,
            height: self.terminal_height,
        };

        // Create layout with conversation on top and input on bottom
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(input_height)])
            .split(size);

        let paragraph = self.render_paragraph(conversation, chunks[0]);

        self.terminal.draw(|f| {
            Self::render_ui_static(f, input_height, input, cursor, paragraph, chunks[0]);
        })?;

        Ok(())
    }

    pub fn resized(&mut self, width: u16, height: u16) {
        log::debug!("Window resized: {width}x{height}");
        self.terminal_width = width;
        self.terminal_height = height;
        self.markdown_renderer
            .with_width(width.saturating_sub(4) as usize);

        self.formatted.clear();
    }

    pub fn handle_scroll_back(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_add(10_u16);
    }

    pub fn handle_scroll_forward(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_sub(10_u16);
    }

    pub fn reset_scroll(&mut self) {
        self.scroll_offset = 0;
    }

    fn calculate_input_height(&self, input_text: &str) -> u16 {
        let available_width = self.terminal_width.saturating_sub(4); // Account for borders

        let lines = if input_text.is_empty() {
            1
        } else {
            input_text
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

        (lines + 2).min(10) // Cap at 10 lines maximum
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

    fn render_paragraph<'a>(
        &mut self,
        conversation: impl IntoIterator<Item = &'a Message>,
        area: ratatui::layout::Rect,
    ) -> Paragraph<'static> {
        let mut lines: VecDeque<Line<'static>> = VecDeque::with_capacity(area.height as usize - 2);
        let mut scroll_offset = self.scroll_offset;

        let rendered_lines = conversation
            .into_iter()
            .filter_map(|msg| {
                let (style, id, content) = match msg {
                    Message::User { id, content, .. } => {
                        (Style::default().fg(Color::Cyan), id, Some(content))
                    }
                    Message::Assistant { id, content, .. } => {
                        (Style::default().fg(Color::Green), id, Some(content))
                    }
                    Message::System { id, content } => {
                        (Style::default().fg(Color::Yellow), id, Some(content))
                    }
                    Message::Thinking { id, content, .. } => (
                        Style::default()
                            .fg(Color::Magenta)
                            .add_modifier(Modifier::ITALIC),
                        id,
                        Some(content),
                    ),
                    Message::ThinkingDone { id, .. } => (Style::default(), id, None),
                    Message::Error { id, content } => {
                        (Style::default().fg(Color::Red), id, Some(content))
                    }
                };

                let rendered_content = if let Some(cached) = self.formatted.get(id) {
                    cached.clone()
                } else {
                    let rendered_content = self
                        .markdown_renderer
                        .render(content?)
                        .lines()
                        .map(str::to_owned)
                        .collect::<Vec<_>>();

                    self.formatted.insert(id.clone(), rendered_content.clone());
                    rendered_content
                };

                // Split the rendered content into lines and apply styling
                let rendered = rendered_content
                    .into_iter()
                    .map(|line| Line::from(vec![Span::styled(line, style)]))
                    .chain(std::iter::once(Line::from("")))
                    .rev()
                    .collect::<Vec<_>>();

                Some(rendered)
            })
            .flatten()
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
        self.scroll_offset -= scroll_offset;

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
