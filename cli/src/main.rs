use clap::{Parser, Subcommand};
use lib::{
    dice, index,
    index::{NoopStore, QdrantStore},
};
use result::Result;

mod result;

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    /// URL of the Qdrant instance to use for vector search (e.g. http://localhost:6334).
    /// When not provided, only the local database is used.
    #[arg(short, long, global = true)]
    qdrant_url: Option<String>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Roll a dice expression using [Caith](https://github.com/Geobert/caith?tab=readme-ov-file#syntax)
    Roll { expr: Vec<String> },

    /// Index the contents of a directory for use as RAG inputs to the AI agent
    Index { path: String },

    /// Run a search against the RAG database for a block of text
    Search {
        text: Vec<String>,
        max_results: Option<u64>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let rt = tokio::runtime::Runtime::new()?;
    let store: Box<dyn index::VectorStore> = match cli.qdrant_url.as_deref() {
        Some(url) => Box::new(QdrantStore::new(url)),
        None => Box::new(NoopStore),
    };
    let index = rt.block_on(index::DocumentIndex::new(store))?;

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
        Command::Index { path } => match rt.block_on(index.index_path(path)) {
            Ok(_) => println!("Finished indexing {path}"),
            Err(e) => println!("Indexing failed: {:?}", e),
        },
        Command::Search { text, max_results } => rt
            .block_on(index.search(&text.join(" "), max_results.unwrap_or(7)))?
            .into_iter()
            .for_each(|r| println!("{r}\n\n")),
    };

    Ok(())
}
