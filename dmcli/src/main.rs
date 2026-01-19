use caith::Roller;
use clap::{Parser, Subcommand};
use result::Result;

mod error;
mod result;

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Roll { expr: Vec<String> },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match execute(&cli.command) {
        Ok(msg) => println!("{}", msg),
        Err(e) => eprintln!("{}", e),
    };

    Ok(())
}

fn execute(command: &Command) -> Result<String> {
    match command {
        Command::Roll { expr } => {
            let expr = expr.join(" ");
            let roller = Roller::new(&expr)?;
            let result = roller.roll()?;
            Ok(format!("🎲 {result}"))
        }
    }
}
