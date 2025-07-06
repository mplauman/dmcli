//! Input handling using crossterm for terminal interaction.
//!
//! This module provides a crossterm-based input handler that replaces the previous
//! rustyline implementation. It offers:
//!
//! - Line editing with cursor movement
//! - Command history navigation
//! - Keyboard shortcuts (Ctrl+A, Ctrl+E, Ctrl+U, etc.)
//! - Non-blocking input with timeout for integration with async event loops
//!
//! ## Key Differences from Rustyline
//!
//! - **Raw terminal mode**: Direct control over terminal input/output
//! - **Non-blocking**: Uses polling with timeout to integrate with async event loops
//! - **Lightweight**: No external readline library dependencies
//! - **Custom key handling**: Full control over keyboard shortcuts and behavior
//!
//! ## Supported Key Bindings
//!
//! - `Enter`: Submit current line
//! - `Ctrl+C`: Exit application
//! - `Backspace`/`Delete`: Character deletion
//! - `Arrow Keys`: Cursor movement and history navigation
//! - `Ctrl+Left`/`Ctrl+Right`: Move cursor by word
//! - `Home`/`End`: Jump to line start/end
//! - `Ctrl+A`/`Ctrl+E`: Jump to line start/end
//! - `Ctrl+U`: Clear entire line
//! - `Ctrl+W`: Delete word backward

use crate::chat_history::ChatHistory;
use crate::commands::{DmCli, DmCommand};
use crate::errors::Error;
use crate::events::AppEvent;
use clap::Parser;
use crossterm::{
    event::{Event, EventStream, KeyCode, KeyEvent, KeyModifiers},
    terminal::{self},
};
use futures::StreamExt;
use shlex::split;

/// Crossterm-based input handler for terminal interaction
pub struct InputHandler {
    event_sender: async_channel::Sender<AppEvent>,
    chat_history: ChatHistory,
    history_index: Option<usize>,
    current_line: String,
    cursor_position: usize,
    event_stream: EventStream,
}

impl InputHandler {
    /// Creates a new input handler and enables raw terminal mode
    pub fn new(event_sender: async_channel::Sender<AppEvent>) -> Result<Self, Error> {
        // Enable raw mode for terminal input
        terminal::enable_raw_mode()?;

        // Create chat history with a temporary directory
        let temp_dir = std::env::temp_dir().join("dmcli_chat_history");
        let chat_history = ChatHistory::new(temp_dir)?;

        Ok(Self {
            event_sender,
            chat_history,
            history_index: None,
            current_line: String::new(),
            cursor_position: 0,
            event_stream: EventStream::new(),
        })
    }

    fn send_event(&self, event: AppEvent) {
        if let Err(e) = self.event_sender.try_send(event) {
            panic!("Failed to send event to UI thread: {e:?}");
        }
    }

    fn input_updated(&self) {
        self.send_event(AppEvent::InputUpdated {
            line: self.current_line.clone(),
            cursor: self.cursor_position,
        });
    }

    /// Attempts to read and process input using event streams
    ///
    /// This method waits for the next event from the terminal and processes it.
    /// It should be called repeatedly in the input worker thread.
    pub async fn read_input(&mut self) {
        let Some(event) = self.event_stream.next().await else {
            return;
        };

        let event = event.unwrap_or_else(|e| {
            panic!("Input event failure: {e:?}");
        });

        match event {
            // Ctrl+C: Exit application
            Event::Key(KeyEvent {
                code: KeyCode::Char('c'),
                modifiers: KeyModifiers::CONTROL,
                ..
            }) => {
                self.send_event(AppEvent::Exit);
            }
            // Enter: Submit current line as command or user input
            Event::Key(KeyEvent {
                code: KeyCode::Enter,
                modifiers: KeyModifiers::NONE,
                ..
            }) => {
                let line = self.current_line.clone();

                self.reset_input_state();

                if !line.is_empty() {
                    self.add_to_history(line.clone());

                    // Parse and send the command/input
                    let event = if let Some(command) = parse_command(&line) {
                        AppEvent::UserCommand(command)
                    } else {
                        AppEvent::UserAgent(line)
                    };

                    self.send_event(event);
                }
            }
            // Shift+Enter: Insert newline character
            Event::Key(KeyEvent {
                code: KeyCode::Enter,
                modifiers: KeyModifiers::SHIFT,
                ..
            }) => {
                self.current_line.insert(self.cursor_position, '\n');
                self.cursor_position += 1;
                self.input_updated();
            }
            // Backspace: Delete character before cursor
            Event::Key(KeyEvent {
                code: KeyCode::Backspace,
                ..
            }) => {
                if self.cursor_position > 0 {
                    self.current_line.remove(self.cursor_position - 1);
                    self.cursor_position -= 1;
                    self.input_updated();
                }
            }
            // Delete: Delete character at cursor
            Event::Key(KeyEvent {
                code: KeyCode::Delete,
                ..
            }) => {
                if self.cursor_position < self.current_line.len() {
                    self.current_line.remove(self.cursor_position);
                    self.input_updated();
                }
            }
            // Left arrow: Move cursor one position left
            Event::Key(KeyEvent {
                code: KeyCode::Left,
                modifiers: KeyModifiers::NONE,
                ..
            }) => {
                if self.cursor_position > 0 {
                    self.cursor_position -= 1;
                    self.input_updated();
                }
            }
            // Right arrow: Move cursor one position right
            Event::Key(KeyEvent {
                code: KeyCode::Right,
                modifiers: KeyModifiers::NONE,
                ..
            }) => {
                if self.cursor_position < self.current_line.len() {
                    self.cursor_position += 1;
                    self.input_updated();
                }
            }
            // Ctrl+Left: Move cursor to previous word boundary
            Event::Key(KeyEvent {
                code: KeyCode::Left,
                modifiers: KeyModifiers::CONTROL,
                ..
            }) => {
                let new_position = self.find_word_boundary_left();
                let move_distance = self.cursor_position - new_position;
                if move_distance > 0 {
                    self.cursor_position = new_position;
                    self.input_updated();
                }
            }
            // Ctrl+Right: Move cursor to next word boundary
            Event::Key(KeyEvent {
                code: KeyCode::Right,
                modifiers: KeyModifiers::CONTROL,
                ..
            }) => {
                let new_position = self.find_word_boundary_right();
                let move_distance = new_position - self.cursor_position;
                if move_distance > 0 {
                    self.cursor_position = new_position;
                    self.input_updated();
                }
            }
            // Up arrow: Navigate to previous history entry
            Event::Key(KeyEvent {
                code: KeyCode::Up, ..
            }) => {
                self.navigate_history(HistoryDirection::Previous);
            }
            // Down arrow: Navigate to next history entry
            Event::Key(KeyEvent {
                code: KeyCode::Down,
                ..
            }) => {
                self.navigate_history(HistoryDirection::Next);
            }
            // Home: Move cursor to beginning of line
            Event::Key(KeyEvent {
                code: KeyCode::Home,
                ..
            }) => {
                self.cursor_position = 0;
                self.input_updated();
            }
            // End: Move cursor to end of line
            Event::Key(KeyEvent {
                code: KeyCode::End, ..
            }) => {
                self.cursor_position = self.current_line.len();
                self.input_updated();
            }
            // Ctrl+A: Move cursor to beginning of line
            Event::Key(KeyEvent {
                code: KeyCode::Char('a'),
                modifiers: KeyModifiers::CONTROL,
                ..
            }) => {
                self.cursor_position = 0;
                self.input_updated();
            }
            // Ctrl+E: Move cursor to end of line
            Event::Key(KeyEvent {
                code: KeyCode::Char('e'),
                modifiers: KeyModifiers::CONTROL,
                ..
            }) => {
                self.cursor_position = self.current_line.len();
                self.input_updated();
            }
            // Ctrl+U: Clear entire line
            Event::Key(KeyEvent {
                code: KeyCode::Char('u'),
                modifiers: KeyModifiers::CONTROL,
                ..
            }) => {
                self.current_line.clear();
                self.cursor_position = 0;
                self.input_updated();
            }
            // Ctrl+W: Delete word backward
            Event::Key(KeyEvent {
                code: KeyCode::Char('w'),
                modifiers: KeyModifiers::CONTROL,
                ..
            }) => {
                self.delete_word_backward();
                self.input_updated();
            }
            // Page Up: Scroll conversation up
            Event::Key(KeyEvent {
                code: KeyCode::PageUp,
                ..
            }) => {
                self.send_event(AppEvent::ScrollBack);
            }
            // Page Down: Scroll conversation down
            Event::Key(KeyEvent {
                code: KeyCode::PageDown,
                ..
            }) => {
                self.send_event(AppEvent::ScrollForward);
            }
            // Regular character input: Insert character at cursor
            Event::Key(KeyEvent {
                code: KeyCode::Char(c),
                modifiers: KeyModifiers::NONE | KeyModifiers::SHIFT,
                ..
            }) => {
                self.current_line.insert(self.cursor_position, c);
                self.cursor_position += 1;
                self.input_updated();
            }
            // Terminal resize: Update window dimensions
            Event::Resize(width, height) => {
                self.send_event(AppEvent::WindowResized { width, height });
            }
            // All other events: Ignore (mouse events, etc.)
            _ => {
                // Ignore other events (like mouse events)
            }
        };
    }

    fn navigate_history(&mut self, direction: HistoryDirection) {
        let recent_messages = self.chat_history.get_recent_messages();
        if recent_messages.is_empty() {
            return;
        }

        match direction {
            HistoryDirection::Previous => {
                match self.history_index {
                    None => {
                        // First time navigating history, go to last item
                        self.history_index = Some(recent_messages.len() - 1);
                        self.current_line = recent_messages[recent_messages.len() - 1].clone();
                    }
                    Some(index) if index > 0 => {
                        // Go to previous item
                        self.history_index = Some(index - 1);
                        self.current_line = recent_messages[index - 1].clone();
                    }
                    Some(_) => {
                        // Already at the beginning, do nothing
                        return;
                    }
                }
            }
            HistoryDirection::Next => {
                match self.history_index {
                    Some(index) if index < recent_messages.len() - 1 => {
                        // Go to next item
                        self.history_index = Some(index + 1);
                        self.current_line = recent_messages[index + 1].clone();
                    }
                    Some(_) => {
                        // At the end, clear current line
                        self.history_index = None;
                        self.current_line.clear();
                    }
                    None => {
                        // No history navigation active, do nothing
                        return;
                    }
                }
            }
        }

        self.cursor_position = self.current_line.len();
        self.input_updated();
    }

    fn add_to_history(&mut self, line: String) {
        // Use the new ChatHistory to add the message
        if let Err(e) = self.chat_history.add_message(line) {
            log::warn!("Failed to add message to chat history: {}", e);
        }
    }

    fn delete_word_backward(&mut self) {
        if self.cursor_position == 0 {
            return;
        }

        let mut pos = self.cursor_position;
        let chars: Vec<char> = self.current_line.chars().collect();

        // Skip whitespace
        while pos > 0 && chars[pos - 1].is_whitespace() {
            pos -= 1;
        }

        // Delete word characters
        while pos > 0 && !chars[pos - 1].is_whitespace() {
            pos -= 1;
        }

        // Remove the characters
        self.current_line.drain(pos..self.cursor_position);
        self.cursor_position = pos;
        self.input_updated();
    }

    fn reset_input_state(&mut self) {
        self.current_line.clear();
        self.cursor_position = 0;
        self.history_index = None;
        self.input_updated();
    }

    /// Find the position of the previous word boundary (for Ctrl+Left)
    fn find_word_boundary_left(&self) -> usize {
        if self.cursor_position == 0 {
            return 0;
        }

        let chars: Vec<char> = self.current_line.chars().collect();
        let mut pos = self.cursor_position;

        // Skip whitespace to the left
        while pos > 0 && chars[pos - 1].is_whitespace() {
            pos -= 1;
        }

        // Skip non-whitespace to the left (the current word)
        while pos > 0 && !chars[pos - 1].is_whitespace() {
            pos -= 1;
        }

        pos
    }

    /// Find the position of the next word boundary (for Ctrl+Right)
    fn find_word_boundary_right(&self) -> usize {
        let chars: Vec<char> = self.current_line.chars().collect();
        let mut pos = self.cursor_position;

        // Skip non-whitespace to the right (the current word)
        while pos < chars.len() && !chars[pos].is_whitespace() {
            pos += 1;
        }

        // Skip whitespace to the right
        while pos < chars.len() && chars[pos].is_whitespace() {
            pos += 1;
        }

        pos
    }
}

impl Drop for InputHandler {
    fn drop(&mut self) {
        // Restore normal terminal mode
        let _ = terminal::disable_raw_mode();
    }
}

/// Direction for history navigation
enum HistoryDirection {
    /// Navigate to previous history entry
    Previous,
    /// Navigate to next history entry
    Next,
}

/// Attempts to parse a line as a command
///
/// Returns `Some(command)` if the line is a valid command, `None` otherwise.
/// Lines that are not commands are treated as agent input.
fn parse_command(line: &str) -> Option<DmCommand> {
    match split(line).map(DmCli::try_parse_from) {
        Some(Ok(DmCli { command })) => Some(command),
        _ => None,
    }
}
