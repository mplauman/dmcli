use clap::{Parser, Subcommand, ValueEnum};
use lib::{
    dice, index,
    index::{NoopStore, SearchResults, SqliteStore},
};
use result::Result;

mod result;

/// Output format for search results.
#[derive(Clone, ValueEnum)]
enum OutputFormat {
    /// Human-readable Markdown document (default).
    Markdown,
    /// XML document blocks suitable for injection into an LLM prompt.
    Xml,
    /// JSON array for structured agent-skill responses.
    Json,
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    /// Path to SQLite database for vector storage (optional)
    #[arg(short, long)]
    db_path: Option<String>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Roll a dice expression using [Caith](https://github.com/Geobert/caith?tab=readme-ov-file#syntax)
    Roll { expr: Vec<String> },

    /// Index the contents of a directory for use as RAG inputs to the AI agent
    Index { path: String },

    /// Search the RAG database for text relevant to a query
    Search {
        /// Output format: markdown (default), xml, or json
        #[arg(short, long, default_value = "markdown")]
        output: OutputFormat,

        /// The query text
        text: Vec<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let rt = tokio::runtime::Runtime::new()?;
    let store: Box<dyn index::VectorStore> = match cli.db_path {
        Some(path) => Box::new(rt.block_on(SqliteStore::new(&path))?),
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
        Command::Search { text, output } => {
            let results = SearchResults::from(rt.block_on(index.search(&text.join(" "), 7))?);
            let formatted = match output {
                OutputFormat::Markdown => results.to_string(),
                OutputFormat::Xml => results.to_xml(),
                OutputFormat::Json => results.to_json()?,
            };
            println!("{formatted}");
        }
    };

    Ok(())
}
