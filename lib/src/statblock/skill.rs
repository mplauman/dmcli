//! Generates an AGY agent skill file from a stat block.
//!
//! The emitted skill instructs an AGY agent to produce DM-ready flavour text
//! for the creature: a one-paragraph read-aloud description and one evocative
//! sentence per feature. The stat block is embedded as XML context so the
//! agent has concrete numbers to reference.

use std::fmt;

use crate::statblock::GeneratedStatBlock;

/// An AGY skill file ready to be written to a `.md` file.
pub struct StatBlockSkill {
    /// Creature name, used in the skill name and instructions.
    name: String,
    /// The generated stat block, embedded as XML context.
    statblock: GeneratedStatBlock,
}

impl StatBlockSkill {
    /// Construct a new [`StatBlockSkill`].
    pub fn new(name: impl Into<String>, statblock: GeneratedStatBlock) -> Self {
        Self {
            name: name.into(),
            statblock,
        }
    }
}

impl fmt::Display for StatBlockSkill {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let xml = self.statblock.to_xml();
        let slug = self
            .name
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join("-");

        writeln!(f, "---")?;
        writeln!(f, "name: {slug}")?;
        writeln!(f, "description: >-")?;
        writeln!(
            f,
            "  Generates DM-ready flavour text for the {} encounter.",
            self.name
        )?;
        writeln!(f, "---")?;
        writeln!(f)?;
        writeln!(f, "# {} Encounter Skill", self.name)?;
        writeln!(f)?;
        writeln!(f, "You are helping a DM run a {} encounter.", self.name)?;
        writeln!(f, "The stat block below contains the creature's mechanics.")?;
        writeln!(f)?;
        writeln!(f, "## Stat Block")?;
        writeln!(f)?;
        writeln!(f, "```xml")?;
        writeln!(f, "{xml}")?;
        writeln!(f, "```")?;
        writeln!(f)?;
        writeln!(f, "## Your Task")?;
        writeln!(f)?;
        writeln!(
            f,
            "When activated, produce the following for the DM. Keep everything \
concise — the DM needs to read this at the table, not study it."
        )?;
        writeln!(f)?;
        writeln!(f, "### 1. Read-Aloud Description")?;
        writeln!(f)?;
        writeln!(
            f,
            "Write one paragraph (3–4 sentences) of vivid, present-tense prose \
the DM can read aloud when the creature is first seen. Focus on appearance, \
movement, and immediate threat. No combat statistics."
        )?;
        writeln!(f)?;
        writeln!(f, "### 2. Feature Flavour")?;
        writeln!(f)?;
        writeln!(
            f,
            "For each `<feature>` in the stat block, write one sentence of \
flavour that explains *how* that ability looks or feels when triggered — \
something the DM can say when the creature uses it. Keep the actual mechanical \
numbers from the stat block visible in parentheses so the DM can reference \
them without switching context."
        )?;
        writeln!(f)?;
        writeln!(f, "### 3. Tactics Note")?;
        writeln!(f)?;
        writeln!(
            f,
            "Write 2–3 bullet points describing how this creature fights: \
when it uses each feature, how it positions itself, and when it retreats or \
presses the attack. Base this on the stat block numbers, not assumptions."
        )?;
        writeln!(f)?;
        writeln!(f, "## Format")?;
        writeln!(f)?;
        writeln!(
            f,
            "Output plain markdown with the three sections above as `##` \
headers. No preamble, no sign-off."
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statblock::{System, generate};

    fn fixed(_expr: &str) -> crate::result::Result<i64> {
        Ok(10)
    }

    fn goblin_block() -> GeneratedStatBlock {
        generate("Goblin", "1/4", System::Dnd5e, None, &[], fixed).expect("generation succeeds")
    }

    #[test]
    fn display_contains_yaml_frontmatter() {
        let skill = StatBlockSkill::new("Goblin", goblin_block());
        let output = skill.to_string();

        assert!(output.starts_with("---\n"), "must open with YAML fence");
        assert!(output.contains("name: goblin"), "slug in frontmatter");
        assert!(
            output.contains("description:"),
            "description in frontmatter"
        );
    }

    #[test]
    fn display_contains_xml_stat_block() {
        let skill = StatBlockSkill::new("Goblin", goblin_block());
        let output = skill.to_string();

        assert!(output.contains("```xml"), "xml fence present");
        assert!(output.contains("<statblock"), "xml content present");
        assert!(output.contains("</statblock>"), "xml content closed");
    }

    #[test]
    fn display_contains_all_three_task_sections() {
        let skill = StatBlockSkill::new("Goblin", goblin_block());
        let output = skill.to_string();

        assert!(output.contains("Read-Aloud Description"));
        assert!(output.contains("Feature Flavour"));
        assert!(output.contains("Tactics Note"));
    }

    #[test]
    fn slug_uses_hyphens_for_spaces() {
        let block = generate("Cave Troll", "5", System::Dnd5e, None, &[], fixed).expect("succeeds");
        let skill = StatBlockSkill::new("Cave Troll", block);
        let output = skill.to_string();

        assert!(output.contains("name: cave-troll"));
    }
}
