use clap::{Parser, Subcommand, ValueEnum};
use lib::dice;
use result::Result;

mod result;

/// Output format for command results.
#[derive(Clone, ValueEnum)]
pub enum OutputFormat {
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
pub struct Cli {
    #[command(subcommand)]
    pub command: CliCommand,
}

#[derive(Subcommand)]
pub enum CliCommand {
    /// Roll a dice expression using [Caith](https://github.com/Geobert/caith?tab=readme-ov-file#syntax)
    Roll {
        /// Output format: markdown (default), xml, or json
        #[arg(short, long, default_value = "markdown")]
        output: OutputFormat,
        expr: Vec<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        CliCommand::Roll { expr, output } => {
            let roll = dice::roll(&expr.join(" "))?;
            let formatted = match output {
                OutputFormat::Markdown => roll.to_string(),
                OutputFormat::Xml => roll.to_xml(),
                OutputFormat::Json => roll.to_json()?,
            };
            println!("{formatted}");
        }
    }

    Ok(())
}
