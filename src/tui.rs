use crate::errors::Error;
use crate::events::AppEvent;
use config::Config;
use crossterm::{
    cursor::{self},
    execute,
    style::{Color, Print, ResetColor, SetForegroundColor},
    terminal::{Clear, ClearType},
};
use std::io::{self, Write};

pub struct Tui {
    current_line: String,
    cursor_position: usize,
}

impl Tui {
    pub fn new(
        _config: &Config,
        _event_sender: async_channel::Sender<AppEvent>,
    ) -> Result<Self, Error> {
        Ok(Self {
            current_line: String::new(),
            cursor_position: 0,
        })
    }

    pub fn render(&mut self) -> Result<(), Error> {
        // Clear current line and redraw
        execute!(
            io::stdout(),
            cursor::MoveToColumn(0),
            Clear(ClearType::CurrentLine),
            SetForegroundColor(Color::Green),
            Print(">> "),
            ResetColor,
            Print(&self.current_line),
            cursor::MoveToColumn((">> ".len() + self.cursor_position) as u16)
        )?;
        io::stdout().flush()?;
        Ok(())
    }

    pub fn append(&mut self, message: &str) {
        // Clear the line and move cursor to start of next line
        execute!(
            io::stdout(),
            cursor::MoveToColumn(0),
            Clear(ClearType::CurrentLine),
            Print('\n')
        )
        .expect("this shouldn't be done in here");

        println!("{}", message);
    }

    pub fn input_updated(&mut self, current_line: String, cursor_position: usize) {
        self.current_line = current_line;
        self.cursor_position = cursor_position;
    }

    pub fn resized(&mut self, _width: u16, _height: u16) {
        log::debug!("Window resized: {}x{}", _width, _height);
    }
}
