//! Configuration and argument parsing for dmcli.
//!
//! [`AppConfig::parse`] is the single entry point: it parses CLI arguments
//! with clap (so `--help` and defaults work normally), loads the TOML config
//! file, and merges the two — CLI flags always win, config file supplies
//! defaults for anything the user did not pass explicitly.
//!
//! # Config file location
//!
//! The default location is `$XDG_CONFIG_HOME/dmcli/config.toml`
//! (`~/.config/dmcli/config.toml` when `XDG_CONFIG_HOME` is unset).
//! Override with `-C <path>` / `--config <path>`.
//!
//! # Example config file
//!
//! ```toml
//! [dmcli]
//! db_path = "/home/user/.local/share/dmcli/chunks.db"
//!
//! [dmcli.index]
//! path = "/home/user/documents/campaign"
//! ```

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand, ValueEnum};
use serde::Deserialize;

use crate::result::{Error, Result};

// ---------------------------------------------------------------------------
// CLI layer (raw clap parse, before config-file merging)
// ---------------------------------------------------------------------------

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
struct Cli {
    /// Path to a TOML configuration file.
    /// Defaults to $XDG_CONFIG_HOME/dmcli/config.toml
    /// (~/.config/dmcli/config.toml when XDG_CONFIG_HOME is not set).
    #[arg(short = 'C', long, global = true)]
    config: Option<PathBuf>,

    /// Path to the SQLite database file used for vector storage.
    /// When not provided, indexing and search are no-ops.
    #[arg(short, long, global = true)]
    db_path: Option<PathBuf>,

    #[command(subcommand)]
    command: CliCommand,
}

#[derive(Subcommand)]
enum CliCommand {
    /// Roll a dice expression using [Caith](https://github.com/Geobert/caith?tab=readme-ov-file#syntax)
    Roll {
        /// Output format: markdown (default), xml, or json
        #[arg(short, long, default_value = "markdown")]
        output: OutputFormat,
        expr: Vec<String>,
    },

    /// Index the contents of a directory for use as RAG inputs to the AI agent.
    /// When no path is given, falls back to dmcli.index.path in the config file.
    Index { path: Option<String> },

    /// Run a search against the RAG database for a block of text
    Search {
        /// Output format: markdown (default), xml, or json
        #[arg(short, long, default_value = "markdown")]
        output: OutputFormat,
        text: Vec<String>,
    },
}

// ---------------------------------------------------------------------------
// Config file layer (TOML deserialization)
// ---------------------------------------------------------------------------

/// Configuration options specific to the `index` command.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct IndexConfig {
    /// Default path to index when none is supplied on the command line.
    path: Option<PathBuf>,
}

/// Command-specific configuration sections, nested under `[dmcli]`.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct CommandsConfig {
    /// Options for the `index` command (`[dmcli.index]`).
    index: IndexConfig,
}

/// Global dmcli configuration options (`[dmcli]`).
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct DmcliConfig {
    /// Path to the SQLite database file.
    db_path: Option<PathBuf>,

    /// Command-specific overrides (e.g. `[dmcli.index]`).
    #[serde(flatten)]
    commands: CommandsConfig,
}

/// Root configuration document — only the `[dmcli]` table is read.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct FileConfig {
    dmcli: DmcliConfig,
}

impl FileConfig {
    /// Load from an explicit path, or from the XDG default location.
    ///
    /// Returns [`FileConfig::default`] (all `None`) if no file exists at the
    /// resolved path, so callers never need to special-case a missing file.
    fn load(path: Option<&Path>) -> Result<Self> {
        let resolved = match path {
            Some(p) => p.to_path_buf(),
            None => default_config_path()?,
        };

        if !resolved.exists() {
            return Ok(Self::default());
        }

        let contents = std::fs::read_to_string(&resolved)
            .map_err(|e| Error::Config(format!("cannot read {}: {e}", resolved.display())))?;

        toml::from_str::<Self>(&contents)
            .map_err(|e| Error::Config(format!("cannot parse {}: {e}", resolved.display())))
    }
}

// ---------------------------------------------------------------------------
// Merged result
// ---------------------------------------------------------------------------

/// The fully-resolved command the user wants to run.
pub enum Command {
    /// Roll a dice expression.
    Roll {
        expr: Vec<String>,
        output: OutputFormat,
    },
    /// Index a directory.
    Index { path: PathBuf },
    /// Search the RAG database.
    Search {
        text: Vec<String>,
        output: OutputFormat,
    },
}

/// The merged result of CLI flags and config file values.
///
/// Obtain one with [`AppConfig::parse`].  CLI flags always take precedence;
/// config-file values supply defaults for anything not passed on the command
/// line.
pub struct AppConfig {
    /// Path to the SQLite database file, if any.
    pub db_path: Option<PathBuf>,
    /// The resolved command to run.
    pub command: Command,
}

impl AppConfig {
    /// Parse CLI arguments and merge with the config file.
    ///
    /// This is the single entry point for all configuration. It:
    ///
    /// 1. Parses CLI arguments with clap (exits on `--help`, `--version`, or
    ///    invalid arguments as normal).
    /// 2. Loads the TOML config file (default or `--config` override).
    /// 3. Merges the two sources (CLI wins).
    /// 4. Resolves the subcommand, filling in config-file defaults where
    ///    needed (e.g. `index --path`).
    pub fn parse() -> Result<Self> {
        let cli = Cli::parse();
        let file = FileConfig::load(cli.config.as_deref())?;

        // CLI flag wins; config file is the fallback.
        let db_path = cli.db_path.or(file.dmcli.db_path);

        let command = match cli.command {
            CliCommand::Roll { expr, output } => Command::Roll { expr, output },

            CliCommand::Index { path } => {
                // Resolve: CLI arg → config file → error
                let resolved = path
                    .as_deref()
                    .map(PathBuf::from)
                    .or(file.dmcli.commands.index.path)
                    .ok_or_else(|| {
                        Error::Config(
                            "no path supplied and dmcli.index.path is not set in the config file"
                                .to_string(),
                        )
                    })?;

                Command::Index { path: resolved }
            }

            CliCommand::Search { text, output } => Command::Search { text, output },
        };

        Ok(Self { db_path, command })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return the default config file path (`$XDG_CONFIG_HOME/dmcli/config.toml`).
fn default_config_path() -> Result<PathBuf> {
    let config_dir = dirs::config_dir()
        .ok_or_else(|| Error::Config("cannot determine config directory".to_string()))?;

    Ok(config_dir.join("dmcli").join("config.toml"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;
    use crate::result::Error;

    fn write_config(contents: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().expect("temp file");
        write!(f, "{contents}").expect("write temp file");
        f
    }

    /// A path that does not exist should yield a default (all-`None`) config.
    #[test]
    fn missing_file_returns_default() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let absent = tmp.path().join("no_such_file.toml");

        let cfg = FileConfig::load(Some(&absent)).expect("load");

        assert!(cfg.dmcli.db_path.is_none());
        assert!(cfg.dmcli.commands.index.path.is_none());
    }

    /// All supported fields parse correctly.
    #[test]
    fn full_config_parses() {
        let f = write_config(
            r#"
[dmcli]
db_path = "/tmp/chunks.db"

[dmcli.index]
path = "/tmp/docs"
"#,
        );

        let cfg = FileConfig::load(Some(f.path())).expect("load");

        assert_eq!(cfg.dmcli.db_path, Some(PathBuf::from("/tmp/chunks.db")));
        assert_eq!(
            cfg.dmcli.commands.index.path,
            Some(PathBuf::from("/tmp/docs"))
        );
    }

    /// Only the [dmcli.index] section present — global fields default to None.
    #[test]
    fn partial_config_index_only() {
        let f = write_config(
            r#"
[dmcli.index]
path = "/srv/campaign"
"#,
        );

        let cfg = FileConfig::load(Some(f.path())).expect("load");

        assert!(cfg.dmcli.db_path.is_none());
        assert_eq!(
            cfg.dmcli.commands.index.path,
            Some(PathBuf::from("/srv/campaign"))
        );
    }

    /// Completely empty file is valid and produces all-default config.
    #[test]
    fn empty_file_returns_default() {
        let f = write_config("");

        let cfg = FileConfig::load(Some(f.path())).expect("load");

        assert!(cfg.dmcli.db_path.is_none());
        assert!(cfg.dmcli.commands.index.path.is_none());
    }

    /// Unknown keys in the TOML are silently ignored (forward-compatibility).
    #[test]
    fn unknown_keys_are_ignored() {
        let f = write_config(
            r#"
[dmcli]
future_flag = true

[other_tool]
whatever = 42
"#,
        );

        let cfg = FileConfig::load(Some(f.path())).expect("load");

        assert!(cfg.dmcli.db_path.is_none());
    }

    /// Malformed TOML returns a Config error.
    #[test]
    fn malformed_toml_returns_error() {
        let f = write_config("this is not [ valid toml !!!");

        let err = FileConfig::load(Some(f.path())).expect_err("should fail");

        assert!(matches!(err, Error::Config(_)));
    }
}
