//! D&D 5e stat block data structures and formatting.

use std::fmt;
use std::fmt::Write as FmtWrite;

use crate::result::Result;
use crate::stats::AbilityScores;

/// An attack available to a 5e creature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attack {
    /// Attack roll bonus (e.g., +3, -1).
    pub bonus: i8,
    /// Damage dice expression (e.g., "1d6+2").
    pub damage_dice: String,
}

impl Attack {
    /// Construct a new [`Attack`].
    pub fn new(bonus: i8, damage_dice: impl Into<String>) -> Self {
        Self {
            bonus,
            damage_dice: damage_dice.into(),
        }
    }
}

/// A creature feature with its name and concrete rules description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Feature {
    /// Feature name (e.g., "Knockdown").
    pub name: String,
    /// Concrete rules description.
    pub description: String,
}

impl Feature {
    /// Construct a new [`Feature`].
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
        }
    }
}

impl fmt::Display for Feature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name, self.description)
    }
}

/// A 5e stat block for use at the DM's table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatBlock5e {
    /// Creature name.
    pub name: String,
    /// Challenge rating (e.g., "1/4", "5").
    pub cr: String,
    /// Armor class.
    pub ac: u8,
    /// Hit points.
    pub hp: u16,
    /// Hit point dice formula (e.g., "3d6", "7d8+14").
    pub hp_dice: String,
    /// Speed in feet (default 30).
    pub speed: u8,
    /// Six ability scores.
    pub scores: AbilityScores,
    /// Attacks available to the creature.
    pub attacks: Vec<Attack>,
    /// Features available to the creature.
    pub features: Vec<Feature>,
    /// Optional CR archetype note (e.g., "Brute").
    pub cr_note: Option<String>,
}

impl fmt::Display for StatBlock5e {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{}  AC {}  HP {} ({})  Spd {} ft  CR {}",
            self.name, self.ac, self.hp, self.hp_dice, self.speed, self.cr
        )?;
        writeln!(f, "  {}", self.scores)?;

        let mut line3 = String::new();
        if let Some(first_atk) = self.attacks.first() {
            write!(
                line3,
                "  {}× Atk {:+} to hit, {}",
                self.attacks.len(),
                first_atk.bonus,
                first_atk.damage_dice
            )
            .expect("writing to String is infallible");
        }

        if self.features.is_empty() {
            write!(f, "{line3}")?;
        } else {
            if !line3.is_empty() {
                writeln!(f, "{line3}")?;
            }
            for (i, feature) in self.features.iter().enumerate() {
                if i + 1 < self.features.len() {
                    writeln!(f, "  {feature}")?;
                } else {
                    write!(f, "  {feature}")?;
                }
            }
        }

        Ok(())
    }
}

impl StatBlock5e {
    /// Render the stat block as an XML element suitable for LLM prompt context.
    pub fn to_xml(&self) -> String {
        let mut buf = String::new();
        writeln!(buf, "<statblock system=\"5e\">").expect("writing to String is infallible");
        writeln!(buf, "  <name>{}</name>", self.name).expect("writing to String is infallible");
        writeln!(buf, "  <cr>{}</cr>", self.cr).expect("writing to String is infallible");
        writeln!(buf, "  <ac>{}</ac>", self.ac).expect("writing to String is infallible");
        writeln!(buf, "  <hp avg=\"{}\" dice=\"{}\"/>", self.hp, self.hp_dice)
            .expect("writing to String is infallible");
        writeln!(buf, "  <speed>{}</speed>", self.speed).expect("writing to String is infallible");
        writeln!(
            buf,
            "  <scores str=\"{}\" dex=\"{}\" con=\"{}\" int=\"{}\" wis=\"{}\" cha=\"{}\"/>",
            self.scores.str.score(),
            self.scores.dex.score(),
            self.scores.con.score(),
            self.scores.int.score(),
            self.scores.wis.score(),
            self.scores.cha.score()
        )
        .expect("writing to String is infallible");

        if let Some(first_atk) = self.attacks.first() {
            writeln!(
                buf,
                "  <attacks count=\"{}\" bonus=\"{:+}\" damage=\"{}\"/>",
                self.attacks.len(),
                first_atk.bonus,
                first_atk.damage_dice
            )
            .expect("writing to String is infallible");
        } else {
            writeln!(buf, "  <attacks count=\"0\" bonus=\"+0\" damage=\"\"/>")
                .expect("writing to String is infallible");
        }

        if !self.features.is_empty() {
            writeln!(buf, "  <features>").expect("writing to String is infallible");
            for feat in &self.features {
                writeln!(
                    buf,
                    "    <feature name=\"{}\">{}</feature>",
                    feat.name, feat.description
                )
                .expect("writing to String is infallible");
            }
            writeln!(buf, "  </features>").expect("writing to String is infallible");
        }

        write!(buf, "</statblock>").expect("writing to String is infallible");

        buf
    }

    /// Serialize the stat block to a pretty-printed JSON string.
    pub fn to_json(&self) -> Result<String> {
        let mut map = serde_json::Map::new();
        map.insert("system".to_string(), serde_json::json!("5e"));
        map.insert("name".to_string(), serde_json::json!(self.name));
        map.insert("cr".to_string(), serde_json::json!(self.cr));
        map.insert("ac".to_string(), serde_json::json!(self.ac));
        map.insert("hp".to_string(), serde_json::json!(self.hp));
        map.insert("hp_dice".to_string(), serde_json::json!(self.hp_dice));
        map.insert("speed".to_string(), serde_json::json!(self.speed));
        map.insert(
            "scores".to_string(),
            serde_json::json!({
                "str": self.scores.str.score(),
                "dex": self.scores.dex.score(),
                "con": self.scores.con.score(),
                "int": self.scores.int.score(),
                "wis": self.scores.wis.score(),
                "cha": self.scores.cha.score(),
            }),
        );
        let attacks_val: Vec<serde_json::Value> = self
            .attacks
            .iter()
            .map(|atk| {
                serde_json::json!({
                    "bonus": atk.bonus,
                    "damage_dice": atk.damage_dice,
                })
            })
            .collect();
        map.insert("attacks".to_string(), serde_json::json!(attacks_val));

        if !self.features.is_empty() {
            let features_val: Vec<serde_json::Value> = self
                .features
                .iter()
                .map(|feat| {
                    serde_json::json!({
                        "name": feat.name,
                        "description": feat.description,
                    })
                })
                .collect();
            map.insert("features".to_string(), serde_json::json!(features_val));
        }

        let value = serde_json::Value::Object(map);
        serde_json::to_string_pretty(&value).map_err(crate::error::Error::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::AbilityScore;

    fn sample_goblin(features: Vec<Feature>) -> StatBlock5e {
        StatBlock5e {
            name: "Goblin".to_string(),
            cr: "1/4".to_string(),
            ac: 11,
            hp: 13,
            hp_dice: "3d6".to_string(),
            speed: 30,
            scores: AbilityScores {
                str: AbilityScore::new(8),
                dex: AbilityScore::new(14),
                con: AbilityScore::new(10),
                int: AbilityScore::new(10),
                wis: AbilityScore::new(8),
                cha: AbilityScore::new(8),
            },
            attacks: vec![Attack::new(3, "1d6+2")],
            features,
            cr_note: None,
        }
    }

    #[test]
    fn display_contains_required_fields() {
        let block = sample_goblin(vec![Feature::new(
            "Knockdown",
            "On hit: target must succeed on a DC 11 STR save or be knocked prone.",
        )]);
        let output = block.to_string();

        assert!(output.contains("Goblin"));
        assert!(output.contains("AC 11"));
        assert!(output.contains("HP 13 (3d6)"));
        assert!(output.contains("Spd 30 ft"));
        assert!(output.contains("CR 1/4"));
    }

    #[test]
    fn display_attack_line_contains_atk_and_mult() {
        let block = sample_goblin(vec![]);
        let output = block.to_string();

        assert!(output.contains("Atk"));
        assert!(output.contains('×'));
        assert!(output.contains("1× Atk +3 to hit, 1d6+2"));
    }

    #[test]
    fn display_attack_line_negative_bonus() {
        let mut block = sample_goblin(vec![]);
        block.attacks = vec![Attack::new(-1, "1d4-1")];
        let output = block.to_string();

        assert!(output.contains("1× Atk -1 to hit, 1d4-1"));
    }

    #[test]
    fn display_features_presence() {
        let with_features = sample_goblin(vec![Feature::new(
            "Knockdown",
            "On hit: target must succeed on a DC 11 STR save or be knocked prone.",
        )]);
        assert!(with_features.to_string().contains("Knockdown:"));

        let without_features = sample_goblin(vec![]);
        assert!(!without_features.to_string().contains("Knockdown"));
    }

    #[test]
    fn display_exact_format_without_features() {
        let block = sample_goblin(vec![]);
        let output = block.to_string();
        let expected = concat!(
            "Goblin  AC 11  HP 13 (3d6)  Spd 30 ft  CR 1/4\n",
            "  STR 8(-1) DEX 14(+2) CON 10(+0) INT 10(+0) WIS 8(-1) CHA 8(-1)\n",
            "  1× Atk +3 to hit, 1d6+2"
        );

        assert_eq!(output, expected);
    }

    #[test]
    fn display_exact_format_with_features() {
        let block = sample_goblin(vec![Feature::new(
            "Knockdown",
            "On hit: target must succeed on a DC 11 STR save or be knocked prone.",
        )]);
        let output = block.to_string();
        let expected = concat!(
            "Goblin  AC 11  HP 13 (3d6)  Spd 30 ft  CR 1/4\n",
            "  STR 8(-1) DEX 14(+2) CON 10(+0) INT 10(+0) WIS 8(-1) CHA 8(-1)\n",
            "  1× Atk +3 to hit, 1d6+2\n",
            "  Knockdown: On hit: target must succeed on a DC 11 STR save or be knocked prone."
        );

        assert_eq!(output, expected);
    }

    #[test]
    fn xml_starts_and_ends_correctly() {
        let block = sample_goblin(vec![Feature::new(
            "Knockdown",
            "On hit: target must succeed on a DC 11 STR save or be knocked prone.",
        )]);
        let xml = block.to_xml();

        assert!(xml.starts_with("<statblock"));
        assert!(xml.ends_with("</statblock>"));
        assert!(xml.contains("<name>Goblin</name>"));
        assert!(xml.contains("<cr>1/4</cr>"));
        assert!(xml.contains("<ac>11</ac>"));
        assert!(xml.contains("<hp avg=\"13\" dice=\"3d6\"/>"));
        assert!(xml.contains("<speed>30</speed>"));
        assert!(
            xml.contains(
                "<scores str=\"8\" dex=\"14\" con=\"10\" int=\"10\" wis=\"8\" cha=\"8\"/>"
            )
        );
        assert!(xml.contains("<attacks count=\"1\" bonus=\"+3\" damage=\"1d6+2\"/>"));
        assert!(xml.contains("<feature name=\"Knockdown\">On hit: target must succeed on a DC 11 STR save or be knocked prone.</feature>"));
    }

    #[test]
    fn xml_omits_features_when_empty() {
        let block = sample_goblin(vec![]);
        let xml = block.to_xml();

        assert!(!xml.contains("<features>"));
    }

    #[test]
    fn json_parses_as_valid_with_system_field() {
        let block = sample_goblin(vec![Feature::new(
            "Knockdown",
            "On hit: target must succeed on a DC 11 STR save or be knocked prone.",
        )]);
        let json_str = block.to_json().expect("JSON serialization succeeds");
        let parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("Valid JSON expected");

        assert_eq!(parsed["system"], "5e");
        assert_eq!(parsed["name"], "Goblin");
        assert_eq!(parsed["cr"], "1/4");
        assert_eq!(parsed["ac"], 11);
        assert_eq!(parsed["hp"], 13);
        assert_eq!(parsed["hp_dice"], "3d6");
        assert_eq!(parsed["speed"], 30);
        assert_eq!(parsed["scores"]["str"], 8);
        assert_eq!(parsed["scores"]["dex"], 14);
        assert_eq!(parsed["scores"]["con"], 10);
        assert_eq!(parsed["scores"]["int"], 10);
        assert_eq!(parsed["scores"]["wis"], 8);
        assert_eq!(parsed["scores"]["cha"], 8);
        assert_eq!(parsed["attacks"][0]["bonus"], 3);
        assert_eq!(parsed["attacks"][0]["damage_dice"], "1d6+2");
        assert_eq!(parsed["features"][0]["name"], "Knockdown");
        assert!(
            parsed["features"][0]["description"]
                .as_str()
                .unwrap()
                .contains("DC 11 STR save")
        );
    }

    #[test]
    fn json_omits_features_when_empty() {
        let block = sample_goblin(vec![]);
        let json_str = block.to_json().expect("JSON serialization succeeds");
        let parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("Valid JSON expected");

        assert!(parsed.get("features").is_none());
    }
}
