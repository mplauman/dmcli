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

use crate::commands::{DmCli, DmCommand};
use crate::errors::Error;
use crate::events::AppEvent;
use clap::Parser;
use crossterm::{
    cursor::{self, MoveLeft, MoveRight, MoveToColumn},
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers, poll},
    execute,
    style::{Color, Print, ResetColor, SetForegroundColor},
    terminal::{self, Clear, ClearType},
};
use shlex::split;
use std::io::{self, Write};
use std::sync::mpsc;
use std::time::Duration;

/// Maximum number of history entries to keep
const MAX_HISTORY: usize = 1000;

/// Input polling timeout in milliseconds
const POLL_TIMEOUT_MS: u64 = 50;

/// Crossterm-based input handler for terminal interaction
pub struct InputHandler {
    event_sender: mpsc::Sender<AppEvent>,
    history: Vec<String>,
    history_index: Option<usize>,
    current_line: String,
    cursor_position: usize,
    prompt: String,
}

impl InputHandler {
    /// Creates a new input handler and enables raw terminal mode
    pub fn new(event_sender: mpsc::Sender<AppEvent>) -> Result<Self, Error> {
        // Enable raw mode for terminal input
        terminal::enable_raw_mode()?;

        Ok(Self {
            event_sender,
            history: Vec::new(),
            history_index: None,
            current_line: String::new(),
            cursor_position: 0,
            prompt: ">> ".to_string(),
        })
    }

    fn send_event(&self, event: AppEvent) {
        if let Err(e) = self.event_sender.send(event) {
            panic!("Failed to send event to UI thread: {:?}", e);
        }
    }

    /// Attempts to read and process input with a timeout
    ///
    /// This method is non-blocking and returns quickly if no input is available.
    /// It should be called repeatedly in the main event loop.
    pub async fn read_input(&mut self) -> Result<(), Error> {
        // Display prompt if this is a fresh input
        if self.current_line.is_empty() && self.cursor_position == 0 {
            self.display_prompt()?;
        }

        // Check for input with a short timeout
        if poll(Duration::from_millis(POLL_TIMEOUT_MS))? {
            if let Event::Key(key_event) = event::read()? {
                match self.handle_key_event(key_event)? {
                    InputAction::Continue => {}
                    InputAction::Submit(line) => {
                        if !line.is_empty() {
                            self.add_to_history(line.clone());
                        }

                        // Clear the line and move cursor to start of next line
                        execute!(
                            io::stdout(),
                            cursor::MoveToColumn(0),
                            Clear(ClearType::CurrentLine),
                            Print('\n')
                        )?;

                        // Parse and send the command/input
                        let event = if let Some(command) = parse_command(&line) {
                            AppEvent::UserCommand(command)
                        } else {
                            AppEvent::UserAgent(line)
                        };

                        self.send_event(event);

                        // Reset for next input
                        self.reset_input_state();
                    }
                    InputAction::Exit => {
                        self.send_event(AppEvent::Exit);
                    }
                }
            }
        }

        Ok(())
    }

    fn handle_key_event(&mut self, key_event: KeyEvent) -> Result<InputAction, Error> {
        match key_event {
            // Ctrl+C - Exit
            KeyEvent {
                code: KeyCode::Char('c'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => Ok(InputAction::Exit),

            // Enter - Submit line
            KeyEvent {
                code: KeyCode::Enter,
                ..
            } => Ok(InputAction::Submit(self.current_line.clone())),

            // Backspace - Delete character before cursor
            KeyEvent {
                code: KeyCode::Backspace,
                ..
            } => {
                if self.cursor_position > 0 {
                    self.current_line.remove(self.cursor_position - 1);
                    self.cursor_position -= 1;
                    self.display_prompt()?;
                }
                Ok(InputAction::Continue)
            }

            // Delete - Delete character at cursor
            KeyEvent {
                code: KeyCode::Delete,
                ..
            } => {
                if self.cursor_position < self.current_line.len() {
                    self.current_line.remove(self.cursor_position);
                    self.display_prompt()?;
                }
                Ok(InputAction::Continue)
            }

            // Left arrow - Move cursor left
            KeyEvent {
                code: KeyCode::Left,
                modifiers: KeyModifiers::NONE,
                ..
            } => {
                if self.cursor_position > 0 {
                    self.cursor_position -= 1;
                    execute!(io::stdout(), MoveLeft(1))?;
                }
                Ok(InputAction::Continue)
            }

            // Right arrow - Move cursor right
            KeyEvent {
                code: KeyCode::Right,
                modifiers: KeyModifiers::NONE,
                ..
            } => {
                if self.cursor_position < self.current_line.len() {
                    self.cursor_position += 1;
                    execute!(io::stdout(), MoveRight(1))?;
                }
                Ok(InputAction::Continue)
            }

            // Ctrl+Left - Move cursor left by word
            KeyEvent {
                code: KeyCode::Left,
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                let new_position = self.find_word_boundary_left();
                let move_distance = self.cursor_position - new_position;
                if move_distance > 0 {
                    self.cursor_position = new_position;
                    execute!(io::stdout(), MoveLeft(move_distance as u16))?;
                }
                Ok(InputAction::Continue)
            }

            // Ctrl+Right - Move cursor right by word
            KeyEvent {
                code: KeyCode::Right,
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                let new_position = self.find_word_boundary_right();
                let move_distance = new_position - self.cursor_position;
                if move_distance > 0 {
                    self.cursor_position = new_position;
                    execute!(io::stdout(), MoveRight(move_distance as u16))?;
                }
                Ok(InputAction::Continue)
            }

            // Up arrow - Previous history
            KeyEvent {
                code: KeyCode::Up, ..
            } => {
                self.navigate_history(HistoryDirection::Previous)?;
                Ok(InputAction::Continue)
            }

            // Down arrow - Next history
            KeyEvent {
                code: KeyCode::Down,
                ..
            } => {
                self.navigate_history(HistoryDirection::Next)?;
                Ok(InputAction::Continue)
            }

            // Home - Move to beginning of line
            KeyEvent {
                code: KeyCode::Home,
                ..
            } => {
                self.cursor_position = 0;
                execute!(io::stdout(), MoveToColumn(self.prompt.len() as u16))?;
                Ok(InputAction::Continue)
            }

            // End - Move to end of line
            KeyEvent {
                code: KeyCode::End, ..
            } => {
                self.cursor_position = self.current_line.len();
                execute!(
                    io::stdout(),
                    MoveToColumn((self.prompt.len() + self.current_line.len()) as u16)
                )?;
                Ok(InputAction::Continue)
            }

            // Ctrl+A - Move to beginning of line
            KeyEvent {
                code: KeyCode::Char('a'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.cursor_position = 0;
                execute!(io::stdout(), MoveToColumn(self.prompt.len() as u16))?;
                Ok(InputAction::Continue)
            }

            // Ctrl+E - Move to end of line
            KeyEvent {
                code: KeyCode::Char('e'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.cursor_position = self.current_line.len();
                execute!(
                    io::stdout(),
                    MoveToColumn((self.prompt.len() + self.current_line.len()) as u16)
                )?;
                Ok(InputAction::Continue)
            }

            // Ctrl+U - Clear line
            KeyEvent {
                code: KeyCode::Char('u'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.current_line.clear();
                self.cursor_position = 0;
                self.display_prompt()?;
                Ok(InputAction::Continue)
            }

            // Ctrl+W - Delete word backward
            KeyEvent {
                code: KeyCode::Char('w'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.delete_word_backward();
                self.display_prompt()?;
                Ok(InputAction::Continue)
            }

            // Regular character input
            KeyEvent {
                code: KeyCode::Char(c),
                modifiers: KeyModifiers::NONE | KeyModifiers::SHIFT,
                ..
            } => {
                self.current_line.insert(self.cursor_position, c);
                self.cursor_position += 1;
                self.display_prompt()?;
                Ok(InputAction::Continue)
            }

            // Ignore other key events
            _ => Ok(InputAction::Continue),
        }
    }

    fn display_prompt(&self) -> Result<(), Error> {
        // Clear current line and redraw
        execute!(
            io::stdout(),
            cursor::MoveToColumn(0),
            Clear(ClearType::CurrentLine),
            SetForegroundColor(Color::Green),
            Print(&self.prompt),
            ResetColor,
            Print(&self.current_line),
            cursor::MoveToColumn((self.prompt.len() + self.cursor_position) as u16)
        )?;
        io::stdout().flush()?;
        Ok(())
    }

    fn navigate_history(&mut self, direction: HistoryDirection) -> Result<(), Error> {
        if self.history.is_empty() {
            return Ok(());
        }

        match direction {
            HistoryDirection::Previous => {
                match self.history_index {
                    None => {
                        // First time navigating history, go to last item
                        self.history_index = Some(self.history.len() - 1);
                        self.current_line = self.history[self.history.len() - 1].clone();
                    }
                    Some(index) if index > 0 => {
                        // Go to previous item
                        self.history_index = Some(index - 1);
                        self.current_line = self.history[index - 1].clone();
                    }
                    Some(_) => {
                        // Already at the beginning, do nothing
                        return Ok(());
                    }
                }
            }
            HistoryDirection::Next => {
                match self.history_index {
                    Some(index) if index < self.history.len() - 1 => {
                        // Go to next item
                        self.history_index = Some(index + 1);
                        self.current_line = self.history[index + 1].clone();
                    }
                    Some(_) => {
                        // At the end, clear current line
                        self.history_index = None;
                        self.current_line.clear();
                    }
                    None => {
                        // No history navigation active, do nothing
                        return Ok(());
                    }
                }
            }
        }

        self.cursor_position = self.current_line.len();
        self.display_prompt()?;
        Ok(())
    }

    fn add_to_history(&mut self, line: String) {
        // Don't add empty lines or duplicates of the last entry
        if line.is_empty() || self.history.last() == Some(&line) {
            return;
        }

        self.history.push(line);

        // Limit history size
        if self.history.len() > MAX_HISTORY {
            self.history.remove(0);
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
    }

    fn reset_input_state(&mut self) {
        self.current_line.clear();
        self.cursor_position = 0;
        self.history_index = None;
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

/// Actions that can result from key event processing
enum InputAction {
    /// Continue input processing
    Continue,
    /// Submit the completed line
    Submit(String),
    /// Exit the application
    Exit,
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
