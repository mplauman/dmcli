use clap::{Parser, Subcommand};
use dmlib::DmlibResult;
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

    let result = match &cli.command {
        Command::Roll { expr } => {
            let expr = expr.join(" ");
            let result = dmlib::dice::roll(&expr)?;
            result
        }
        Command::Index { path, sync } => dmlib::index::index(path.as_str(), *sync)?,
        Command::Search { text } => rt.block_on(db.search(&text.join(" "), u64::MAX))?,
    };

    match result {
        DmlibResult::SingleDiceRoll(value, Some(reason)) => println!("🎲 {value} ({reason}"),
        DmlibResult::SingleDiceRoll(value, None) => println!("🎲 {value}"),
        DmlibResult::MultiDiceRoll(values, Some(reason)) => println!("🎲🎲 {values:?} ({reason})"),
        DmlibResult::MultiDiceRoll(values, None) => println!("🎲🎲 {values:?}"),
        DmlibResult::AsyncIndexResult(path) => println!("Running async index on {path}"),
        DmlibResult::IndexResult(path) => println!("Finished indexing {path}"),
        DmlibResult::SearchResult(texts) => println!("Finshed search: {texts:?}"),
    }

    Ok(())
}
