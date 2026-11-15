//! Emits a reusable AGY agent skill for generating DM-ready stat block flavour text.
//!
//! The skill instructs the agent to run `dmcli statblock` for a creature the
//! user describes, then produce a read-aloud description, per-feature flavour
//! sentences, and a tactics note — all from the XML output of that command.

/// The static text of the AGY skill file.
pub const SKILL_TEXT: &str = include_str!("skill.md");
