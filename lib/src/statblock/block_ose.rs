//! Old-School Essentials (OSE) stat block data structures and formatting.

use std::fmt;
use std::fmt::Write as FmtWrite;

use crate::result::Result;

/// Old-School Essentials alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OseAlignment {
    /// Lawful alignment.
    Lawful,
    /// Neutral alignment.
    Neutral,
    /// Chaotic alignment.
    Chaotic,
}

impl fmt::Display for OseAlignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OseAlignment::Lawful => write!(f, "L"),
            OseAlignment::Neutral => write!(f, "N"),
            OseAlignment::Chaotic => write!(f, "C"),
        }
    }
}

/// An attack available to an OSE creature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OseAttack {
    /// Number of attacks (e.g., 1).
    pub count: u8,
    /// Weapon or attack type (e.g., "weapon").
    pub weapon: String,
    /// Damage dice expression (e.g., "1d6").
    pub dice: String,
}

impl OseAttack {
    /// Construct a new [`OseAttack`].
    pub fn new(count: u8, weapon: impl Into<String>, dice: impl Into<String>) -> Self {
        Self {
            count,
            weapon: weapon.into(),
            dice: dice.into(),
        }
    }
}

/// An Old-School Essentials stat block for use at the DM's table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatBlockOse {
    /// Creature name.
    pub name: String,
    /// Descending armor class (traditional B/X value).
    pub ac_descending: u8,
    /// Ascending armor class.
    pub ac_ascending: u8,
    /// Hit dice formula (e.g., "1-1", "2", "3+1").
    pub hd: String,
    /// Hit points.
    pub hp: u8,
    /// Attacks available to the creature.
    pub attacks: Vec<OseAttack>,
    /// Movement rate in feet.
    pub mv_ft: u16,
    /// Encounter movement rate in feet.
    pub mv_ft_enc: u16,
    /// Morale score (2-12).
    pub morale: u8,
    /// Alignment.
    pub alignment: OseAlignment,
}

impl fmt::Display for StatBlockOse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let atks = self
            .attacks
            .iter()
            .map(|a| format!("{}×{} ({})", a.count, a.weapon, a.dice))
            .collect::<Vec<_>>()
            .join(", ");
        write!(
            f,
            "{}  AC {}[{}]  HD {}  hp {}  Atk {}  MV {}'({}')  ML {}  AL {}",
            self.name,
            self.ac_descending,
            self.ac_ascending,
            self.hd,
            self.hp,
            atks,
            self.mv_ft,
            self.mv_ft_enc,
            self.morale,
            self.alignment
        )
    }
}

impl StatBlockOse {
    /// Render the stat block as an XML element suitable for LLM prompt context.
    pub fn to_xml(&self) -> String {
        let mut buf = String::new();
        writeln!(buf, "<statblock system=\"ose\">").expect("writing to String is infallible");
        writeln!(buf, "  <name>{}</name>", self.name).expect("writing to String is infallible");
        writeln!(
            buf,
            "  <ac descending=\"{}\" ascending=\"{}\"/>",
            self.ac_descending, self.ac_ascending
        )
        .expect("writing to String is infallible");
        writeln!(buf, "  <hd>{}</hd>", self.hd).expect("writing to String is infallible");
        writeln!(buf, "  <hp>{}</hp>", self.hp).expect("writing to String is infallible");

        if let Some(first_atk) = self.attacks.first() {
            writeln!(
                buf,
                "  <attacks count=\"{}\" weapon=\"{}\" dice=\"{}\"/>",
                first_atk.count, first_atk.weapon, first_atk.dice
            )
            .expect("writing to String is infallible");
        } else {
            writeln!(buf, "  <attacks count=\"0\" weapon=\"\" dice=\"\"/>")
                .expect("writing to String is infallible");
        }

        writeln!(
            buf,
            "  <mv feet=\"{}\" enc=\"{}\"/>",
            self.mv_ft, self.mv_ft_enc
        )
        .expect("writing to String is infallible");
        writeln!(buf, "  <morale>{}</morale>", self.morale)
            .expect("writing to String is infallible");
        writeln!(buf, "  <alignment>{}</alignment>", self.alignment)
            .expect("writing to String is infallible");
        write!(buf, "</statblock>").expect("writing to String is infallible");

        buf
    }

    /// Serialize the stat block to a pretty-printed JSON string.
    pub fn to_json(&self) -> Result<String> {
        let mut map = serde_json::Map::new();
        map.insert("system".to_string(), serde_json::json!("ose"));
        map.insert("name".to_string(), serde_json::json!(self.name));
        map.insert(
            "ac_descending".to_string(),
            serde_json::json!(self.ac_descending),
        );
        map.insert(
            "ac_ascending".to_string(),
            serde_json::json!(self.ac_ascending),
        );
        map.insert("hd".to_string(), serde_json::json!(self.hd));
        map.insert("hp".to_string(), serde_json::json!(self.hp));
        let atks_json: Vec<_> = self
            .attacks
            .iter()
            .map(|a| {
                serde_json::json!({
                    "count": a.count,
                    "weapon": a.weapon,
                    "dice": a.dice,
                })
            })
            .collect();
        map.insert("attacks".to_string(), serde_json::json!(atks_json));
        map.insert("mv_ft".to_string(), serde_json::json!(self.mv_ft));
        map.insert("mv_ft_enc".to_string(), serde_json::json!(self.mv_ft_enc));
        map.insert("morale".to_string(), serde_json::json!(self.morale));
        map.insert(
            "alignment".to_string(),
            serde_json::json!(self.alignment.to_string()),
        );

        let value = serde_json::Value::Object(map);
        serde_json::to_string_pretty(&value).map_err(crate::error::Error::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_goblin() -> StatBlockOse {
        StatBlockOse {
            name: "Goblin".to_string(),
            ac_descending: 6,
            ac_ascending: 13,
            hd: "1-1".to_string(),
            hp: 4,
            attacks: vec![OseAttack::new(1, "weapon", "1d6")],
            mv_ft: 60,
            mv_ft_enc: 20,
            morale: 7,
            alignment: OseAlignment::Chaotic,
        }
    }

    #[test]
    fn display_contains_required_fields() {
        let block = sample_goblin();
        let output = block.to_string();

        assert!(output.contains("AC"));
        assert!(output.contains("HD"));
        assert!(output.contains("hp"));
        assert!(output.contains("Atk"));
        assert!(output.contains("MV"));
        assert!(output.contains("ML"));
        assert!(output.contains("AL"));
    }

    #[test]
    fn display_exact_one_line_format() {
        let block = sample_goblin();
        let output = block.to_string();
        let expected =
            "Goblin  AC 6[13]  HD 1-1  hp 4  Atk 1×weapon (1d6)  MV 60'(20')  ML 7  AL C";

        assert_eq!(output, expected);
    }

    #[test]
    fn xml_starts_and_ends_correctly() {
        let block = sample_goblin();
        let xml = block.to_xml();

        assert!(xml.starts_with("<statblock"));
        assert!(xml.ends_with("</statblock>"));
        assert!(xml.contains("<name>Goblin</name>"));
        assert!(xml.contains("<ac descending=\"6\" ascending=\"13\"/>"));
        assert!(xml.contains("<hd>1-1</hd>"));
        assert!(xml.contains("<hp>4</hp>"));
        assert!(xml.contains("<attacks count=\"1\" weapon=\"weapon\" dice=\"1d6\"/>"));
        assert!(xml.contains("<mv feet=\"60\" enc=\"20\"/>"));
        assert!(xml.contains("<morale>7</morale>"));
        assert!(xml.contains("<alignment>C</alignment>"));
    }

    #[test]
    fn json_parses_as_valid() {
        let block = sample_goblin();
        let json_str = block.to_json().expect("JSON serialization succeeds");
        let parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("Valid JSON expected");

        assert_eq!(parsed["system"], "ose");
        assert_eq!(parsed["name"], "Goblin");
        assert_eq!(parsed["ac_descending"], 6);
        assert_eq!(parsed["ac_ascending"], 13);
        assert_eq!(parsed["hd"], "1-1");
        assert_eq!(parsed["hp"], 4);
        assert_eq!(parsed["attacks"][0]["count"], 1);
        assert_eq!(parsed["attacks"][0]["weapon"], "weapon");
        assert_eq!(parsed["attacks"][0]["dice"], "1d6");
        assert_eq!(parsed["mv_ft"], 60);
        assert_eq!(parsed["mv_ft_enc"], 20);
        assert_eq!(parsed["morale"], 7);
        assert_eq!(parsed["alignment"], "C");
    }
}
