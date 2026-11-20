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

/// Monster feature for 5e stat block generation.
#[derive(Clone, Copy, ValueEnum)]
pub enum StatblockFeature {
    /// Single-target ranged attack using standard attack bonus and damage.
    DamagingBlast,
    /// Melee attacker takes damage equal to half one attack; costs 1 attack.
    DamageReflection,
    /// Bonus action: teleport up to 30 ft to a visible unoccupied space.
    MistyStep,
    /// On hit: target must succeed on STR save or be knocked prone.
    Knockdown,
    /// On hit: target is grappled and restrained.
    RestrainingGrab,
    /// Action: 10-ft burst anywhere within 120 ft; half DPR.
    DamagingBurst,
    /// Bonus action: Dash, Disengage, or Hide.
    CunningAction,
    /// Creatures starting turn within 10 ft take half one attack damage; costs 1 attack.
    DamagingAura,
    /// Weapon attacks deal additional elemental/magical damage.
    EnergyWeapons,
    /// On taking damage: transfer half or all damage to a willing creature within 60 ft.
    DamageTransference,
}

impl StatblockFeature {
    /// Return the CLI-friendly slug representation.
    pub fn slug(&self) -> &'static str {
        match self {
            StatblockFeature::DamagingBlast => "damaging-blast",
            StatblockFeature::DamageReflection => "damage-reflection",
            StatblockFeature::MistyStep => "misty-step",
            StatblockFeature::Knockdown => "knockdown",
            StatblockFeature::RestrainingGrab => "restraining-grab",
            StatblockFeature::DamagingBurst => "damaging-burst",
            StatblockFeature::CunningAction => "cunning-action",
            StatblockFeature::DamagingAura => "damaging-aura",
            StatblockFeature::EnergyWeapons => "energy-weapons",
            StatblockFeature::DamageTransference => "damage-transference",
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
        feature: Vec<StatblockFeature>,

        /// Output format [default: markdown]
        #[arg(short, long, default_value = "markdown")]
        output: OutputFormat,

        /// Monster name (quote multi-word names)
        name: String,

        /// CR for 5e ("0", "1/8", "1/4", "1/2", "1"–"30"), level for Shadowdark ("1"–"20"), or HD string for OSE ("2", "1-1", "3+1")
        cr_or_level: String,
    },
    /// Emit an AGY agent skill file for the statblock command to stdout
    ///
    /// Pipe the output to a SKILL.md inside a skill directory. When activated,
    /// the agent runs `dmcli statblock` for a creature of the user's choosing
    /// and enriches the output with a read-aloud description, per-feature
    /// flavour sentences, and a tactics note.
    ///
    /// Example:
    ///   dmcli statblock-skill > ~/.gemini/skills/statblock-flavour/SKILL.md
    #[command(name = "statblock-skill")]
    StatblockSkill,
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
            let feature_strs: Vec<&str> = feature.iter().map(StatblockFeature::slug).collect();
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
        CliCommand::StatblockSkill => {
            print!("{}", lib::statblock::SKILL_TEXT);
        }
    }

    Ok(())
}
