//! Emits a reusable AGY agent skill for generating DM-ready stat block flavour text.
//!
//! The skill instructs the agent to run `dmcli statblock` for a creature the
//! user describes, then produce a read-aloud description, per-feature flavour
//! sentences, and a tactics note — all from the XML output of that command.

/// The static text of the AGY skill file.
pub const SKILL_TEXT: &str = include_str!("skill.md");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skill_text_has_frontmatter_and_instructions() {
        let mut lines = SKILL_TEXT.lines();
        assert_eq!(lines.next(), Some("---"));
        assert_eq!(lines.next(), Some("name: statblock-flavour"));
        assert!(SKILL_TEXT.contains("description:"));
        assert!(SKILL_TEXT.contains("dmcli statblock"));
        assert!(SKILL_TEXT.contains("Look & Feel"));
    }
}
