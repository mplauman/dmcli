use clap::Parser;
use shlex::split;

use crate::commands::{DmCli, DmCommand};
use crate::errors::Error;

mod commands;
mod errors;

fn read_line(readline: &mut rustyline::DefaultEditor) -> Result<Option<String>, Error> {
    match readline.readline(">> ") {
        Ok(line) => Ok(Some(line)),
        Err(rustyline::error::ReadlineError::Interrupted) => Ok(None),
        Err(x) => Err(x.into()),
    }
}

fn parse(line: &str) -> Option<DmCommand> {
    match split(line).map(DmCli::try_parse_from) {
        Some(Ok(DmCli { command })) => Some(command),
        _ => None,
    }
}

fn main() -> Result<(), Error> {
    let mut rl = rustyline::DefaultEditor::new().unwrap();

    loop {
        let Some(line) = read_line(&mut rl)? else {
            break;
        };

        if line.is_empty() {
            continue;
        }

        rl.add_history_entry(line.as_str())?;

        // Check to see if the entered text can be handled with one of the built in commands. This
        // is much faster, cheaper, and more accurate than feeding it to an AI agent for processing.
        if let Some(command) = parse(line.as_str()) {
            match command {
                DmCommand::Exit {} => {
                    println!("Good bye!");
                    break;
                }
                DmCommand::Roll { expressions } => {
                    println!("Rolling {:?}", expressions.join(" "));
                }
            }

            continue;
        }

        // Anything that isn't specifically a command just gets fed to the AI agent. This can take
        // a lot longer to deal with (and is more expensive) but is where the magic like "create an
        // NPC with blue hair" type stuff works.
        //
        // (until the agent actually works, just spit out the line)
        println!(">>> {}", line);
    }

    Ok(())
}
