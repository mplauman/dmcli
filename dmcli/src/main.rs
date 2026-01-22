use clap::{Parser, Subcommand};
use dmlib::{dice, index};
use result::Result;

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
    /// Roll a dice expression using [Caith](https://github.com/Geobert/caith?tab=readme-ov-file#syntax)
    Roll { expr: Vec<String> },

    /// Index the contents of a directory for use as RAG inputs to the AI agent
    Index {
        path: String,

        /// Perform the indexing synchronously (this may take a while)
        #[arg(short, long)]
        sync: bool,
    },

    /// Run a search against the RAG database for a block of text
    Search { text: Vec<String> },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let rt = tokio::runtime::Runtime::new()?;
    let db = rt.block_on(dmlib::database::Database::new())?;

    match &cli.command {
        Command::Roll { expr } => match dice::roll(&expr.join(" "))? {
            dice::DiceRoll::Single(value, reason) => println!(
                "🎲 {value}{}",
                reason.map(|r| format!(" ({r})")).unwrap_or_default()
            ),
            dice::DiceRoll::Multi(values, reason) => println!(
                "🎲🎲 {values:?}{}",
                reason.map(|r| format!(" ({r})")).unwrap_or_default()
            ),
        },
        Command::Index { path, sync } => match index::index(path.as_str(), *sync)? {
            index::IndexStatus::Complete(path) => println!("Finished indexing {path}"),
            index::IndexStatus::InProgress(err) => println!("Error indexing {path}: {err}"),
        },
        Command::Search { text } => rt
            .block_on(db.search(&text.join(" "), u64::MAX))?
            .into_iter()
            .for_each(|r| println!("{r}")),
    };

    Ok(())
}
