use crate::commands::{DmCli, DmCommand};
use crate::errors::Error;
use crate::events::AppEvent;
use clap::Parser;
use rustyline::{DefaultEditor, error::ReadlineError};
use shlex::split;
use std::sync::mpsc;

pub struct InputHandler {
    editor: DefaultEditor,
    event_sender: mpsc::Sender<AppEvent>,
}

impl InputHandler {
    pub fn new(event_sender: mpsc::Sender<AppEvent>) -> Result<Self, Error> {
        let editor = DefaultEditor::new()?;
        Ok(Self {
            editor,
            event_sender,
        })
    }

    fn send_event(&self, event: AppEvent) {
        if let Err(e) = self.event_sender.send(event) {
            panic!("Failed to send event to UI thread: {:?}", e);
        }
    }

    pub async fn read_input(&mut self) -> Result<(), Error> {
        match self.editor.readline(">> ") {
            Ok(line) => {
                if !line.is_empty() {
                    self.editor.add_history_entry(line.as_str())?;
                }

                // Try to parse as a command first
                let event = if let Some(command) = parse_command(&line) {
                    AppEvent::UserCommand(command)
                } else {
                    // If not a command, treat as agent input
                    AppEvent::UserAgent(line)
                };

                self.send_event(event);
            }
            Err(ReadlineError::Interrupted) => self.send_event(AppEvent::Exit),
            Err(e) => return Err(e.into()),
        }

        Ok(())
    }
}

fn parse_command(line: &str) -> Option<DmCommand> {
    match split(line).map(DmCli::try_parse_from) {
        Some(Ok(DmCli { command })) => Some(command),
        _ => None,
    }
}
