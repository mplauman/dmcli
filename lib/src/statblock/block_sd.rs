//! Shadowdark stat block data structures and formatting.

use std::fmt;
use std::fmt::Write as FmtWrite;

use crate::result::Result;

/// Shadowdark alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdAlignment {
    /// Lawful alignment.
    Law,
    /// Neutral alignment.
    Neutral,
    /// Chaotic alignment.
    Chaos,
}

impl fmt::Display for SdAlignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SdAlignment::Law => write!(f, "Law"),
            SdAlignment::Neutral => write!(f, "Neutral"),
            SdAlignment::Chaos => write!(f, "Chaos"),
        }
    }
}

/// An attack available to a Shadowdark creature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SdAttack {
    /// Attack or weapon name (e.g., "dagger").
    pub name: String,
    /// Attack roll bonus.
    pub bonus: i8,
    /// Damage dice expression (e.g., "1d4").
    pub dice: String,
}

impl SdAttack {
    /// Construct a new [`SdAttack`].
    pub fn new(name: impl Into<String>, bonus: i8, dice: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            bonus,
            dice: dice.into(),
        }
    }
}

/// A Shadowdark stat block for use at the DM's table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatBlockSd {
    /// Creature name.
    pub name: String,
    /// Armor class.
    pub ac: u8,
    /// Hit points.
    pub hp: u8,
    /// Attacks available to the creature.
    pub attacks: Vec<SdAttack>,
    /// Movement distance ("near", "close", "far").
    pub mv: String,
    /// Alignment.
    pub alignment: SdAlignment,
    /// Level / Challenge rating.
    pub level: u8,
}

impl fmt::Display for StatBlockSd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let atks = self
            .attacks
            .iter()
            .map(|a| format!("{} {:+} ({})", a.name, a.bonus, a.dice))
            .collect::<Vec<_>>()
            .join(", ");
        write!(
            f,
            "{}  AC {}  HP {}  Atk {} | MV {} | AL {} | LVL {}",
            self.name, self.ac, self.hp, atks, self.mv, self.alignment, self.level
        )
    }
}

impl StatBlockSd {
    /// Render the stat block as an XML element suitable for LLM prompt context.
    pub fn to_xml(&self) -> String {
        let mut buf = String::new();
        writeln!(buf, "<statblock system=\"shadowdark\">")
            .expect("writing to String is infallible");
        writeln!(buf, "  <name>{}</name>", self.name).expect("writing to String is infallible");
        writeln!(buf, "  <ac>{}</ac>", self.ac).expect("writing to String is infallible");
        writeln!(buf, "  <hp>{}</hp>", self.hp).expect("writing to String is infallible");

        if let Some(first_atk) = self.attacks.first() {
            writeln!(
                buf,
                "  <attacks count=\"{}\" name=\"{}\" bonus=\"{:+}\" dice=\"{}\"/>",
                self.attacks.len(),
                first_atk.name,
                first_atk.bonus,
                first_atk.dice
            )
            .expect("writing to String is infallible");
        } else {
            writeln!(
                buf,
                "  <attacks count=\"0\" name=\"\" bonus=\"+0\" dice=\"\"/>"
            )
            .expect("writing to String is infallible");
        }

        writeln!(buf, "  <mv>{}</mv>", self.mv).expect("writing to String is infallible");
        writeln!(buf, "  <alignment>{}</alignment>", self.alignment)
            .expect("writing to String is infallible");
        writeln!(buf, "  <level>{}</level>", self.level).expect("writing to String is infallible");
        write!(buf, "</statblock>").expect("writing to String is infallible");

        buf
    }

    /// Serialize the stat block to a pretty-printed JSON string.
    pub fn to_json(&self) -> Result<String> {
        let mut map = serde_json::Map::new();
        map.insert("system".to_string(), serde_json::json!("shadowdark"));
        map.insert("name".to_string(), serde_json::json!(self.name));
        map.insert("ac".to_string(), serde_json::json!(self.ac));
        map.insert("hp".to_string(), serde_json::json!(self.hp));
        let atks_json: Vec<_> = self
            .attacks
            .iter()
            .map(|a| {
                serde_json::json!({
                    "name": a.name,
                    "bonus": a.bonus,
                    "dice": a.dice,
                })
            })
            .collect();
        map.insert("attacks".to_string(), serde_json::json!(atks_json));
        map.insert("mv".to_string(), serde_json::json!(self.mv));
        map.insert(
            "alignment".to_string(),
            serde_json::json!(self.alignment.to_string()),
        );
        map.insert("level".to_string(), serde_json::json!(self.level));

        let value = serde_json::Value::Object(map);
        serde_json::to_string_pretty(&value).map_err(crate::error::Error::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_goblin() -> StatBlockSd {
        StatBlockSd {
            name: "Goblin".to_string(),
            ac: 13,
            hp: 4,
            attacks: vec![SdAttack::new("dagger", 1, "1d4")],
            mv: "near".to_string(),
            alignment: SdAlignment::Chaos,
            level: 1,
        }
    }

    #[test]
    fn display_contains_required_fields() {
        let block = sample_goblin();
        let output = block.to_string();

        assert!(output.contains("AC"));
        assert!(output.contains("HP"));
        assert!(output.contains("Atk"));
        assert!(output.contains("MV"));
        assert!(output.contains("AL"));
        assert!(output.contains("LVL"));
    }

    #[test]
    fn display_exact_one_line_format() {
        let block = sample_goblin();
        let output = block.to_string();
        let expected = "Goblin  AC 13  HP 4  Atk dagger +1 (1d4) | MV near | AL Chaos | LVL 1";

        assert_eq!(output, expected);
    }

    #[test]
    fn xml_starts_and_ends_correctly() {
        let block = sample_goblin();
        let xml = block.to_xml();

        assert!(xml.starts_with("<statblock"));
        assert!(xml.ends_with("</statblock>"));
        assert!(xml.contains("<name>Goblin</name>"));
        assert!(xml.contains("<ac>13</ac>"));
        assert!(xml.contains("<hp>4</hp>"));
        assert!(xml.contains("<mv>near</mv>"));
        assert!(xml.contains("<alignment>Chaos</alignment>"));
        assert!(xml.contains("<level>1</level>"));
    }

    #[test]
    fn json_parses_as_valid() {
        let block = sample_goblin();
        let json_str = block.to_json().expect("JSON serialization succeeds");
        let parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("Valid JSON expected");

        assert_eq!(parsed["system"], "shadowdark");
        assert_eq!(parsed["name"], "Goblin");
        assert_eq!(parsed["ac"], 13);
        assert_eq!(parsed["hp"], 4);
        assert_eq!(parsed["attacks"][0]["name"], "dagger");
        assert_eq!(parsed["attacks"][0]["bonus"], 1);
        assert_eq!(parsed["attacks"][0]["dice"], "1d4");
        assert_eq!(parsed["mv"], "near");
        assert_eq!(parsed["alignment"], "Chaos");
        assert_eq!(parsed["level"], 1);
    }
}
