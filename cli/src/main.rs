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

/// Target tabletop RPG system for stat block generation.
#[derive(Clone, Copy, ValueEnum)]
pub enum StatblockSystem {
    /// Dungeons & Dragons 5th Edition.
    #[value(name = "5e")]
    Dnd5e,
    /// Shadowdark RPG.
    #[value(name = "shadowdark")]
    Shadowdark,
    /// Old-School Essentials (B/X).
    #[value(name = "ose")]
    Ose,
}

impl From<StatblockSystem> for lib::statblock::System {
    fn from(s: StatblockSystem) -> Self {
        match s {
            StatblockSystem::Dnd5e => lib::statblock::System::Dnd5e,
            StatblockSystem::Shadowdark => lib::statblock::System::Shadowdark,
            StatblockSystem::Ose => lib::statblock::System::Ose,
        }
    }
}

/// Combat archetype for 5e stat block generation.
#[derive(Clone, Copy, ValueEnum)]
pub enum StatblockRole {
    /// Minion archetype (CR 1/8 baseline).
    Minion,
    /// Soldier archetype (CR 1/2 baseline).
    Soldier,
    /// Brute archetype (CR 2 baseline, default).
    Brute,
    /// Specialist archetype (CR 4 baseline).
    Specialist,
    /// Myrmidon archetype (CR 7 baseline).
    Myrmidon,
    /// Sentinel archetype (CR 11 baseline).
    Sentinel,
    /// Champion archetype (CR 15 baseline).
    Champion,
}

impl From<StatblockRole> for lib::statblock::Role {
    fn from(r: StatblockRole) -> Self {
        match r {
            StatblockRole::Minion => lib::statblock::Role::Minion,
            StatblockRole::Soldier => lib::statblock::Role::Soldier,
            StatblockRole::Brute => lib::statblock::Role::Brute,
            StatblockRole::Specialist => lib::statblock::Role::Specialist,
            StatblockRole::Myrmidon => lib::statblock::Role::Myrmidon,
            StatblockRole::Sentinel => lib::statblock::Role::Sentinel,
            StatblockRole::Champion => lib::statblock::Role::Champion,
        }
    }
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
    /// Generate a stat block for 5e, Shadowdark, or OSE
    Statblock {
        /// Rule system [default: 5e]
        #[arg(short, long, default_value = "5e")]
        system: StatblockSystem,

        /// Ability score archetype for 5e [default: brute]
        #[arg(short, long, default_value = "brute")]
        role: StatblockRole,

        /// Monster feature slug, repeatable [default: none]
        #[arg(short, long)]
        feature: Vec<String>,

        /// Output format [default: markdown]
        #[arg(short, long, default_value = "markdown")]
        output: OutputFormat,

        /// Monster name (quote multi-word names)
        name: String,

        /// CR for 5e ("0", "1/8", "1/4", "1/2", "1"–"30"), level for Shadowdark ("1"–"20"), or HD string for OSE ("2", "1-1", "3+1")
        cr_or_level: String,
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
        CliCommand::Statblock {
            system,
            role,
            feature,
            output,
            name,
            cr_or_level,
        } => {
            let feature_strs: Vec<&str> = feature.iter().map(String::as_str).collect();
            let statblock = lib::statblock::generate(
                &name,
                &cr_or_level,
                system.into(),
                Some(role.into()),
                &feature_strs,
                |expr| dice::roll(expr).map(|r| r.total()),
            )?;
            let formatted = match output {
                OutputFormat::Markdown => statblock.to_string(),
                OutputFormat::Xml => statblock.to_xml(),
                OutputFormat::Json => statblock.to_json()?,
            };
            println!("{formatted}");
        }
    }

    Ok(())
}
