use clap::Parser;
use shlex::split;

use crate::commands::{DmCli, DmCommand};
use crate::errors::Error;

mod commands;
mod errors;

fn main() -> Result<(), Error> {
    let mut rl = rustyline::DefaultEditor::new().unwrap();

    loop {
        let line = match rl.readline(">> ") {
            Ok(line) => line,
            Err(rustyline::error::ReadlineError::Interrupted) => {
                println!("Interrupted!");
                break;
            }
            Err(x) => {
                return Err(x.into());
            }
        };

        if line.is_empty() {
            continue;
        }

        rl.add_history_entry(line.as_str())?;

        if let Some(Ok(cli)) = split(line.as_str()).map(DmCli::try_parse_from) {
            // They've entered some text that matches one of the configured commands. Run that directly since they'll know exactly what it is they want.
            match cli.command {
                DmCommand::Exit {} => {
                    println!("Good bye!");
                    break;
                }
                DmCommand::Roll { expressions } => {
                    println!("Rolling {:?}", expressions.join(" "));
                }
            }
        } else {
            // They've entered some kind of handwavy text. Fire that off to the AI agent to figure
            // out.
            println!(">>> {}", line);
        }
    }

    Ok(())
}
