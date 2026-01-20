use clap::{Parser, Subcommand};
use dmlib::DmlibResult;
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

    let result = match &cli.command {
        Command::Roll { expr } => {
            let expr = expr.join(" ");
            let result = dmlib::dice::roll(&expr)?;
            result
        }
    };

    match result {
        DmlibResult::SingleDiceRoll(value, Some(reason)) => println!("🎲 {value} ({reason}"),
        DmlibResult::SingleDiceRoll(value, None) => println!("🎲 {value}"),
        DmlibResult::MultiDiceRoll(values, Some(reason)) => println!("🎲🎲 {values:?} ({reason})"),
        DmlibResult::MultiDiceRoll(values, None) => println!("🎲🎲 {values:?}"),
    }

    Ok(())
}
