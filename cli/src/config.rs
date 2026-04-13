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

use std::fs;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand, ValueEnum};
use serde::Deserialize;

use crate::result::{Error, Result};

// ---------------------------------------------------------------------------
// Config schema — single source of truth for keys, sections, and comments
// ---------------------------------------------------------------------------

/// A single configurable key within a TOML section.
struct SchemaKey {
    /// TOML key name (snake_case).
    key: &'static str,
    /// Human-readable description rendered as a `#` comment line.
    comment: &'static str,
    /// The literal TOML value to write (e.g. `"http://localhost:6334"`).
    /// `None` means the key is written commented-out with no value.
    default_value: Option<&'static str>,
    /// Whether to write this key commented-out even when a default is present.
    commented_out: bool,
}

/// A TOML section (table header) plus its keys.
struct SchemaSection {
    /// Dotted TOML table path, e.g. `"dmcli"` or `"dmcli.index"`.
    table: &'static str,
    keys: &'static [SchemaKey],
}

/// The full config schema, ordered as they should appear in the file.
static SCHEMA: &[SchemaSection] = &[
    SchemaSection {
        table: "dmcli",
        keys: &[SchemaKey {
            key: "db_path",
            comment: "Path to the SQLite database file used for vector storage.\n# Default: not set (search and index are no-ops without this)",
            default_value: None,
            commented_out: true,
        }],
    },
    SchemaSection {
        table: "dmcli.index",
        keys: &[SchemaKey {
            key: "path",
            comment: "Default directory to index when none is supplied on the command line.",
            default_value: None,
            commented_out: true,
        }],
    },
];

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

    /// Manage the dmcli configuration file
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
}

#[derive(Subcommand)]
pub enum ConfigAction {
    /// Write a default configuration file.
    ///
    /// Writes to $XDG_CONFIG_HOME/dmcli/config.toml (or --config path).
    /// Fails if the file already exists unless --force is given.
    Init {
        /// Overwrite an existing config file.
        #[arg(long)]
        force: bool,
    },

    /// Add any keys that are missing from the existing configuration file.
    ///
    /// New keys are inserted into the correct section in alphabetical order.
    /// Existing keys and formatting are left untouched.
    Update,
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
    /// Manage the config file.
    Config { action: ConfigAction },
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
    /// The config file path supplied via `--config`, or `None` for the default.
    pub config_path: Option<PathBuf>,
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

            CliCommand::Config { action } => Command::Config { action },
        };

        Ok(Self {
            db_path,
            command,
            config_path: cli.config,
        })
    }
}

// ---------------------------------------------------------------------------
// Config init / update
// ---------------------------------------------------------------------------

/// Execute a `config` subcommand action, returning a human-readable message.
pub fn run_config_action(action: ConfigAction, path: Option<&Path>) -> Result<String> {
    let resolved = match path {
        Some(p) => p.to_path_buf(),
        None => default_config_path()?,
    };

    match action {
        ConfigAction::Init { force } => config_init(&resolved, force),
        ConfigAction::Update => config_update(&resolved),
    }
}

/// Write a fresh default config file.
fn config_init(path: &Path, force: bool) -> Result<String> {
    if path.exists() && !force {
        return Err(Error::Config(format!(
            "{} already exists — use --force to overwrite",
            path.display()
        )));
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| Error::Config(format!("cannot create {}: {e}", parent.display())))?;
    }

    fs::write(path, render_schema())
        .map_err(|e| Error::Config(format!("cannot write {}: {e}", path.display())))?;

    Ok(format!("Created {}", path.display()))
}

/// Add any keys missing from an existing config file, preserving all existing
/// content and formatting.
fn config_update(path: &Path) -> Result<String> {
    if !path.exists() {
        return config_init(path, false);
    }

    let raw = fs::read_to_string(path)
        .map_err(|e| Error::Config(format!("cannot read {}: {e}", path.display())))?;

    let mut doc: toml_edit::DocumentMut = raw
        .parse()
        .map_err(|e| Error::Config(format!("cannot parse {}: {e}", path.display())))?;

    let mut added: Vec<String> = Vec::new();

    for section in SCHEMA {
        // Walk the dotted table path, creating tables as needed.
        let parts: Vec<&str> = section.table.split('.').collect();
        let table = ensure_table(&mut doc, &parts);

        // Collect existing keys so we can detect what is missing.
        // Sort the schema keys and insert missing ones alphabetically.
        let mut section_keys: Vec<&SchemaKey> = section.keys.iter().collect();
        section_keys.sort_by_key(|k| k.key);

        for schema_key in section_keys {
            // Only live (non-commented-out) keys with a default value can be
            // inserted by update — purely commented-out keys exist only in
            // the init template.
            if schema_key.commented_out || schema_key.default_value.is_none() {
                continue;
            }

            if table.contains_key(schema_key.key) {
                continue;
            }

            // Find the alphabetical insertion position among current keys.
            let mut existing_keys: Vec<String> = table.iter().map(|(k, _)| k.to_string()).collect();
            existing_keys.push(schema_key.key.to_string());
            existing_keys.sort();
            let insert_pos = existing_keys
                .iter()
                .position(|k| k == schema_key.key)
                .unwrap_or(existing_keys.len().saturating_sub(1));

            insert_live_key(table, schema_key, insert_pos);
            added.push(format!("{}.{}", section.table, schema_key.key));
        }
    }

    fs::write(path, doc.to_string())
        .map_err(|e| Error::Config(format!("cannot write {}: {e}", path.display())))?;

    if added.is_empty() {
        Ok(format!("{} is already up to date", path.display()))
    } else {
        Ok(format!(
            "Updated {} — added: {}",
            path.display(),
            added.join(", ")
        ))
    }
}

/// Navigate (or create) a chain of dotted table keys within a document,
/// returning a mutable reference to the innermost table.
fn ensure_table<'a>(
    doc: &'a mut toml_edit::DocumentMut,
    parts: &[&str],
) -> &'a mut toml_edit::Table {
    let mut current = doc.as_table_mut();
    for &part in parts {
        // Insert a table if the key is absent.
        if !current.contains_key(part) {
            current.insert(part, toml_edit::Item::Table(toml_edit::Table::new()));
        }
        current = current
            .get_mut(part)
            .and_then(|item| item.as_table_mut())
            .expect("just inserted a table");
    }
    current
}

/// Insert a live (non-commented-out) schema key into a table at a given
/// position, with its description rendered as a `#` comment above it.
///
/// Only keys that have a `default_value` and are not `commented_out` are
/// inserted — purely commented-out keys exist only in the `init` template
/// and are never added by `update`.
fn insert_live_key(table: &mut toml_edit::Table, schema_key: &SchemaKey, position: usize) {
    let raw = match schema_key.default_value {
        Some(v) if !schema_key.commented_out => v,
        // Nothing live to insert.
        _ => return,
    };

    // Parse the raw TOML value string so toml_edit owns it properly.
    let parsed: toml_edit::Value = raw.parse().unwrap_or_else(|_| toml_edit::Value::from(raw));

    let comment_prefix: String = schema_key
        .comment
        .lines()
        .map(|l| format!("# {l}\n"))
        .collect();

    let mut val = parsed;
    val.decor_mut().set_prefix(comment_prefix);

    let item = toml_edit::Item::Value(val);

    // toml_edit 0.22 has no positional insert on Table, so we rebuild order
    // by collecting all pairs, splicing, clearing, and re-inserting.
    let mut pairs: Vec<(String, toml_edit::Item)> = table
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect();

    pairs.insert(position, (schema_key.key.to_string(), item));

    let keys: Vec<String> = pairs.iter().map(|(k, _)| k.clone()).collect();
    for k in &keys {
        table.remove(k);
    }
    for (k, v) in pairs {
        table.insert(&k, v);
    }
}

/// Render the full default config as a TOML string with every key commented
/// out and annotated.
fn render_schema() -> String {
    let mut out = String::new();
    for section in SCHEMA {
        out.push_str(&format!("[{}]\n", section.table));
        let mut keys: Vec<&SchemaKey> = section.keys.iter().collect();
        keys.sort_by_key(|k| k.key);
        for key in keys {
            for line in key.comment.lines() {
                out.push_str(&format!("# {line}\n"));
            }
            match (key.default_value, key.commented_out) {
                (Some(v), true) => out.push_str(&format!("# {} = {v}\n", key.key)),
                (Some(v), false) => out.push_str(&format!("{} = {v}\n", key.key)),
                (None, _) => out.push_str(&format!("# {} =\n", key.key)),
            }
            out.push('\n');
        }
    }
    out
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

    // -----------------------------------------------------------------------
    // config init
    // -----------------------------------------------------------------------

    /// init writes a file that is valid TOML.
    #[test]
    fn init_writes_valid_toml() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let path = tmp.path().join("config.toml");

        let msg = config_init(&path, false).expect("init");
        assert!(msg.contains("Created"));

        let contents = std::fs::read_to_string(&path).expect("read");
        toml::from_str::<toml::Value>(&contents).expect("valid TOML");
    }

    /// init creates parent directories if they don't exist.
    #[test]
    fn init_creates_parent_dirs() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let path = tmp.path().join("nested").join("dirs").join("config.toml");

        config_init(&path, false).expect("init");
        assert!(path.exists());
    }

    /// init fails without --force when the file already exists.
    #[test]
    fn init_fails_if_file_exists() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let path = tmp.path().join("config.toml");

        config_init(&path, false).expect("first init");
        let err = config_init(&path, false).expect_err("second init should fail");
        assert!(matches!(err, Error::Config(_)));
    }

    /// init succeeds with --force when the file already exists.
    #[test]
    fn init_force_overwrites_existing() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let path = tmp.path().join("config.toml");

        config_init(&path, false).expect("first init");
        config_init(&path, true).expect("forced overwrite");
    }

    /// The generated file contains all expected section headers.
    #[test]
    fn init_output_contains_expected_sections() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let path = tmp.path().join("config.toml");

        config_init(&path, false).expect("init");
        let contents = std::fs::read_to_string(&path).expect("read");

        assert!(contents.contains("[dmcli]"));
        assert!(contents.contains("[dmcli.index]"));
    }

    /// All schema keys appear in the generated file (commented out).
    #[test]
    fn init_output_contains_all_schema_keys() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let path = tmp.path().join("config.toml");

        config_init(&path, false).expect("init");
        let contents = std::fs::read_to_string(&path).expect("read");

        for section in SCHEMA {
            for key in section.keys {
                assert!(
                    contents.contains(key.key),
                    "expected key '{}' in generated config",
                    key.key
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // config update
    // -----------------------------------------------------------------------

    /// update on a missing file creates it (same as init).
    #[test]
    fn update_missing_file_creates_it() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let path = tmp.path().join("config.toml");

        let msg = config_update(&path).expect("update");
        assert!(msg.contains("Created"));
        assert!(path.exists());
    }

    /// update on an already-complete file reports no changes.
    #[test]
    fn update_up_to_date_file_reports_no_changes() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let path = tmp.path().join("config.toml");

        // Start from the init output, which contains every key.
        config_init(&path, false).expect("init");
        let msg = config_update(&path).expect("update");
        assert!(
            msg.contains("up to date"),
            "expected 'up to date', got: {msg}"
        );
    }

    /// update preserves existing values and only adds missing live keys.
    #[test]
    fn update_preserves_existing_content() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let path = tmp.path().join("config.toml");

        let initial = "[dmcli]\ndb_path = \"/srv/campaign/chunks.db\"\n";
        std::fs::write(&path, initial).expect("write");

        config_update(&path).expect("update");

        let contents = std::fs::read_to_string(&path).expect("read");
        // The user's value must be preserved verbatim.
        assert!(
            contents.contains("\"/srv/campaign/chunks.db\""),
            "existing value was clobbered: {contents}"
        );
    }

    /// update on a file with malformed TOML returns an error.
    #[test]
    fn update_malformed_toml_returns_error() {
        let f = write_config("this is not [ valid toml !!!");

        let err = config_update(f.path()).expect_err("should fail");
        assert!(matches!(err, Error::Config(_)));
    }

    /// update output is valid TOML.
    #[test]
    fn update_output_is_valid_toml() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let path = tmp.path().join("config.toml");

        // Start with a minimal file missing all keys.
        std::fs::write(&path, "[dmcli]\n").expect("write");
        config_update(&path).expect("update");

        let contents = std::fs::read_to_string(&path).expect("read");
        toml::from_str::<toml::Value>(&contents).expect("valid TOML");
    }
}
