use crate::commands::{DmCli, DmCommand};
use crate::errors::Error;
use crate::events::AppEvent;
use clap::Parser;
use rustyline::{DefaultEditor, error::ReadlineError};
use shlex::split;

pub struct InputHandler {
    editor: DefaultEditor,
}

impl InputHandler {
    pub fn new() -> Result<Self, Error> {
        let editor = DefaultEditor::new()?;
        Ok(Self { editor })
    }

    pub fn read_input(&mut self) -> Result<AppEvent, Error> {
        match self.editor.readline(">> ") {
            Ok(line) => {
                if !line.is_empty() {
                    self.editor.add_history_entry(line.as_str())?;

                    // Try to parse as a command first
                    if let Some(command) = self.parse_command(&line) {
                        Ok(AppEvent::UserCommand(command))
                    } else {
                        // If not a command, treat as agent input
                        Ok(AppEvent::UserAgent(line))
                    }
                } else {
                    // For empty lines, read again
                    self.read_input()
                }
            }
            Err(ReadlineError::Interrupted) => Ok(AppEvent::Exit),
            Err(e) => Err(e.into()),
        }
    }

    fn parse_command(&self, line: &str) -> Option<DmCommand> {
        match split(line).map(DmCli::try_parse_from) {
            Some(Ok(DmCli { command })) => Some(command),
            _ => None,
        }
    }
}

impl Default for InputHandler {
    fn default() -> Self {
        Self::new().expect("Failed to create InputHandler")
    }
}
