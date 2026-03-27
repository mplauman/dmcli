use lib::dice;
use lib::index::{DocumentIndex, NoopStore, SearchResults, SqliteStore, VectorStore};
use result::Result;
use std::process;

mod config;
mod result;

fn main() -> Result<()> {
    let config = config::AppConfig::parse()?;

    // Handle config subcommands before building the index — they don't need
    // the Qdrant connection or the embedding model.
    if let config::Command::Config { action } = config.command {
        match config::run_config_action(action, config.config_path.as_deref()) {
            Ok(msg) => {
                println!("{msg}");
                return Ok(());
            }
            Err(e) => {
                eprintln!("error: {e}");
                process::exit(1);
            }
        }
    }

    let rt = tokio::runtime::Runtime::new()?;
    let store: Box<dyn VectorStore> = match config.db_path {
        Some(path) => Box::new(rt.block_on(SqliteStore::new(path))?),
        None => Box::new(NoopStore),
    };
    let index = rt.block_on(DocumentIndex::new(store))?;

    match config.command {
        config::Command::Config { .. } => unreachable!("handled above"),
        config::Command::Roll { expr, output } => {
            let roll = dice::roll(&expr.join(" "))?;
            let formatted = match output {
                config::OutputFormat::Markdown => roll.to_string(),
                config::OutputFormat::Xml => roll.to_xml(),
                config::OutputFormat::Json => roll.to_json()?,
            };
            println!("{formatted}");
        }
        config::Command::Index { path } => match rt.block_on(index.index_path(&path)) {
            Ok(_) => println!("Finished indexing {}", path.display()),
            Err(e) => println!("Indexing failed: {e}"),
        },
        config::Command::Search { text, output } => {
            let results = SearchResults::from(rt.block_on(index.search(&text.join(" "), 7))?);
            let formatted = match output {
                config::OutputFormat::Markdown => results.to_string(),
                config::OutputFormat::Xml => results.to_xml(),
                config::OutputFormat::Json => results.to_json()?,
            };
            println!("{formatted}");
        }
    }

    Ok(())
}
