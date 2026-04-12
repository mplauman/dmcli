use clap::{Parser, Subcommand, ValueEnum};
use lib::{
    dice, index,
    index::{NoopStore, SearchResults, SqliteStore},
};
use result::Result;

mod result;

/// Output format for command results.
#[derive(Clone, ValueEnum)]
enum OutputFormat {
    /// Human-readable Markdown (default).
    Markdown,
    /// XML element suitable for injection into an LLM prompt.
    Xml,
    /// JSON object or array for structured agent-skill responses.
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
    Roll {
        /// Output format: markdown (default), xml, or json
        #[arg(short, long, default_value = "markdown")]
        output: OutputFormat,

        /// The dice expression to roll (e.g. "2d6 + 3")
        expr: Vec<String>,
    },

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
        Command::Roll { expr, output } => {
            let roll = dice::roll(&expr.join(" "))?;
            let formatted = match output {
                OutputFormat::Markdown => roll.to_string(),
                OutputFormat::Xml => roll.to_xml(),
                OutputFormat::Json => roll.to_json()?,
            };
            println!("{formatted}");
        }
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
