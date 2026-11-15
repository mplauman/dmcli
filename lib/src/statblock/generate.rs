//! Stat block generation logic for D&D 5e, Shadowdark, and OSE.

use std::fmt;

use crate::error::Error;
use crate::result::Result;
use crate::statblock::block_5e::{Attack, StatBlock5e};
use crate::statblock::block_ose::{OseAlignment, OseAttack, StatBlockOse};
use crate::statblock::block_sd::{SdAlignment, SdAttack, StatBlockSd};
use crate::statblock::lgmrd::{ARCHETYPES, CR_TABLE, FEATURES};
use crate::stats::{AbilityScore, AbilityScores};

/// Target tabletop RPG system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum System {
    /// Dungeons & Dragons 5th Edition.
    Dnd5e,
    /// Shadowdark RPG.
    Shadowdark,
    /// Old-School Essentials (B/X).
    Ose,
}

/// Combat archetype — determines ability score array for 5e generation.
/// Defaults to `Brute` when not specified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
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

/// A fully generated stat block across any supported game system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeneratedStatBlock {
    /// D&D 5e stat block.
    Dnd5e(StatBlock5e),
    /// Shadowdark stat block.
    Shadowdark(StatBlockSd),
    /// Old-School Essentials stat block.
    Ose(StatBlockOse),
}

impl fmt::Display for GeneratedStatBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GeneratedStatBlock::Dnd5e(b) => write!(f, "{b}"),
            GeneratedStatBlock::Shadowdark(b) => write!(f, "{b}"),
            GeneratedStatBlock::Ose(b) => write!(f, "{b}"),
        }
    }
}

impl GeneratedStatBlock {
    /// Render the stat block as an XML element.
    pub fn to_xml(&self) -> String {
        match self {
            GeneratedStatBlock::Dnd5e(b) => b.to_xml(),
            GeneratedStatBlock::Shadowdark(b) => b.to_xml(),
            GeneratedStatBlock::Ose(b) => b.to_xml(),
        }
    }

    /// Serialize the stat block to a pretty-printed JSON string.
    pub fn to_json(&self) -> Result<String> {
        match self {
            GeneratedStatBlock::Dnd5e(b) => b.to_json(),
            GeneratedStatBlock::Shadowdark(b) => b.to_json(),
            GeneratedStatBlock::Ose(b) => b.to_json(),
        }
    }
}

fn parse_ose_hd(s: &str) -> Result<u8> {
    let base_part = s.split(['-', '+']).next().unwrap_or("").trim();
    if base_part.is_empty() {
        return Err(Error::StatBlock(format!("invalid OSE HD: {s}")));
    }
    match base_part.parse::<u8>() {
        Ok(n) if n > 0 => Ok(n),
        _ => Err(Error::StatBlock(format!("invalid OSE HD: {s}"))),
    }
}

/// Generate a stat block for the requested system and parameters.
///
/// `features` is a slice of slugs (e.g. `"knockdown"`) matched
/// case-insensitively against `lgmrd::FEATURES`. Unknown slugs return
/// `Err`. Features with `costs_attack = true` reduce attack count by 1,
/// with a floor of 1.
///
/// HP is rolled using `roll_fn`, which accepts a dice expression string
/// and returns `i64`.
pub fn generate(
    name: &str,
    cr_or_level: &str,
    system: System,
    role: Option<Role>,
    features: &[&str],
    roll_fn: impl Fn(&str) -> Result<i64>,
) -> Result<GeneratedStatBlock> {
    match system {
        System::Dnd5e => {
            let cr_row = CR_TABLE
                .iter()
                .find(|row| row.cr == cr_or_level)
                .ok_or_else(|| Error::StatBlock(format!("unknown 5e CR: {cr_or_level}")))?;

            let role = role.unwrap_or(Role::Brute);
            let archetype_name = match role {
                Role::Minion => "Minion",
                Role::Soldier => "Soldier",
                Role::Brute => "Brute",
                Role::Specialist => "Specialist",
                Role::Myrmidon => "Myrmidon",
                Role::Sentinel => "Sentinel",
                Role::Champion => "Champion",
            };
            let archetype = ARCHETYPES
                .iter()
                .find(|a| a.name == archetype_name)
                .expect("archetype must exist in ARCHETYPES");

            let rolled_hp = roll_fn(cr_row.hp_dice)?;
            let hp = rolled_hp.clamp(1, u16::MAX as i64) as u16;

            let mut feature_names = Vec::new();
            let mut cost_attacks = 0u8;
            for slug in features {
                let feat = FEATURES
                    .iter()
                    .find(|f| f.slug.eq_ignore_ascii_case(slug))
                    .ok_or_else(|| Error::StatBlock(format!("unknown feature slug: {slug}")))?;
                feature_names.push(feat.name.to_string());
                if feat.costs_attack {
                    cost_attacks = cost_attacks.saturating_add(1);
                }
            }

            let final_attacks = cr_row.num_attacks.saturating_sub(cost_attacks).max(1);
            let attacks =
                vec![Attack::new(cr_row.prof_bonus, cr_row.damage_dice); final_attacks as usize];

            let scores = AbilityScores {
                str: AbilityScore::new(archetype.str as i8),
                dex: AbilityScore::new(archetype.dex as i8),
                con: AbilityScore::new(archetype.con as i8),
                int: AbilityScore::new(archetype.int as i8),
                wis: AbilityScore::new(archetype.wis as i8),
                cha: AbilityScore::new(archetype.cha as i8),
            };

            let block = StatBlock5e {
                name: name.to_string(),
                cr: cr_row.cr.to_string(),
                ac: cr_row.ac,
                hp,
                hp_dice: cr_row.hp_dice.to_string(),
                speed: 30,
                scores,
                attacks,
                features: feature_names,
                cr_note: Some(archetype.name.to_string()),
            };

            Ok(GeneratedStatBlock::Dnd5e(block))
        }
        System::Shadowdark => {
            let level = cr_or_level.parse::<u8>().map_err(|_| {
                Error::StatBlock(format!(
                    "invalid Shadowdark level: {cr_or_level} (must be 1-20)"
                ))
            })?;
            if !(1..=20).contains(&level) {
                return Err(Error::StatBlock(format!(
                    "invalid Shadowdark level: {level} (must be 1-20)"
                )));
            }

            let rolled_hp = roll_fn(&format!("{level}d6"))?;
            let hp = rolled_hp.clamp(1, u8::MAX as i64) as u8;

            let block = StatBlockSd {
                name: name.to_string(),
                ac: 10,
                hp,
                attacks: vec![SdAttack::new("attack", level as i8, "1d6")],
                mv: "near".to_string(),
                alignment: SdAlignment::Neutral,
                level,
            };

            Ok(GeneratedStatBlock::Shadowdark(block))
        }
        System::Ose => {
            let n = parse_ose_hd(cr_or_level)?;
            let rolled_hp = roll_fn(&format!("{n}d8"))?;
            let hp = rolled_hp.clamp(1, u8::MAX as i64) as u8;
            let ac_ascending = 10 + n;
            let ac_descending = 19 - ac_ascending;

            let block = StatBlockOse {
                name: name.to_string(),
                ac_descending,
                ac_ascending,
                hd: cr_or_level.to_string(),
                hp,
                attacks: vec![OseAttack::new(1, "weapon", "1d6")],
                mv_ft: 60,
                mv_ft_enc: 20,
                morale: 7,
                alignment: OseAlignment::Neutral,
            };

            Ok(GeneratedStatBlock::Ose(block))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed(_expr: &str) -> Result<i64> {
        Ok(10)
    }

    #[test]
    fn dnd5e_goblin_defaults() {
        let block = generate("Goblin", "1/4", System::Dnd5e, None, &[], fixed)
            .expect("generation succeeds");
        if let GeneratedStatBlock::Dnd5e(b) = block {
            assert_eq!(b.hp, 10);
            assert_eq!(b.ac, 11);
            assert_eq!(b.attacks.len(), 1);
            assert_eq!(b.attacks[0].bonus, 3);
            assert!(b.features.is_empty());
        } else {
            panic!("expected Dnd5e variant");
        }
    }

    #[test]
    fn dnd5e_features_slug_case_insensitive() {
        let block = generate("Goblin", "1/4", System::Dnd5e, None, &["knockdown"], fixed)
            .expect("generation succeeds");
        if let GeneratedStatBlock::Dnd5e(b) = block {
            assert_eq!(b.features, vec!["Knockdown"]);
            assert_eq!(b.attacks.len(), 1);
        } else {
            panic!("expected Dnd5e variant");
        }
    }

    #[test]
    fn dnd5e_feature_costs_attack_reduces_count() {
        // CR 1 has 2 attacks
        let normal =
            generate("Orc", "1", System::Dnd5e, None, &[], fixed).expect("generation succeeds");
        if let GeneratedStatBlock::Dnd5e(b) = normal {
            assert_eq!(b.attacks.len(), 2);
        }

        let with_feature = generate(
            "Orc",
            "1",
            System::Dnd5e,
            None,
            &["damage-reflection"],
            fixed,
        )
        .expect("generation succeeds");
        if let GeneratedStatBlock::Dnd5e(b) = with_feature {
            assert_eq!(b.attacks.len(), 1);
        }
    }

    #[test]
    fn dnd5e_costs_attack_floor_is_one() {
        let block = generate(
            "Goblin",
            "1/4",
            System::Dnd5e,
            None,
            &["damage-reflection", "damaging-aura"],
            fixed,
        )
        .expect("generation succeeds");
        if let GeneratedStatBlock::Dnd5e(b) = block {
            assert_eq!(b.attacks.len(), 1);
        }
    }

    #[test]
    fn dnd5e_unknown_feature_slug_err() {
        let res = generate(
            "Goblin",
            "1/4",
            System::Dnd5e,
            None,
            &["unknown-slug"],
            fixed,
        );
        assert!(res.is_err());
    }

    #[test]
    fn dnd5e_unknown_cr_err() {
        let res = generate("Goblin", "99", System::Dnd5e, None, &[], fixed);
        assert!(res.is_err());
    }

    #[test]
    fn shadowdark_level_3() {
        let block = generate("Goblin", "3", System::Shadowdark, None, &[], fixed)
            .expect("generation succeeds");
        if let GeneratedStatBlock::Shadowdark(b) = block {
            assert_eq!(b.hp, 10);
            assert_eq!(b.ac, 10);
            assert_eq!(b.level, 3);
        } else {
            panic!("expected Shadowdark variant");
        }
    }

    #[test]
    fn shadowdark_invalid_level_err() {
        let res = generate("Goblin", "0", System::Shadowdark, None, &[], fixed);
        assert!(res.is_err());
        let res2 = generate("Goblin", "21", System::Shadowdark, None, &[], fixed);
        assert!(res2.is_err());
    }

    #[test]
    fn ose_hd_2() {
        let block =
            generate("Goblin", "2", System::Ose, None, &[], fixed).expect("generation succeeds");
        if let GeneratedStatBlock::Ose(b) = block {
            assert_eq!(b.hp, 10);
            assert_eq!(b.ac_ascending, 12);
            assert_eq!(b.hd, "2");
        } else {
            panic!("expected Ose variant");
        }
    }

    #[test]
    fn ose_hd_1_minus_1() {
        let block =
            generate("Goblin", "1-1", System::Ose, None, &[], fixed).expect("generation succeeds");
        if let GeneratedStatBlock::Ose(b) = block {
            assert_eq!(b.ac_ascending, 11);
            assert_eq!(b.hd, "1-1");
        } else {
            panic!("expected Ose variant");
        }
    }

    #[test]
    fn ose_invalid_hd_err() {
        let res = generate("Goblin", "invalid", System::Ose, None, &[], fixed);
        assert!(res.is_err());
    }

    #[test]
    fn generated_statblock_delegates_non_empty() {
        let dnd = generate("Goblin", "1/4", System::Dnd5e, None, &[], fixed).unwrap();
        assert!(!dnd.to_string().is_empty());
        assert!(!dnd.to_xml().is_empty());
        assert!(!dnd.to_json().unwrap().is_empty());

        let sd = generate("Goblin", "1", System::Shadowdark, None, &[], fixed).unwrap();
        assert!(!sd.to_string().is_empty());
        assert!(!sd.to_xml().is_empty());
        assert!(!sd.to_json().unwrap().is_empty());

        let ose = generate("Goblin", "1-1", System::Ose, None, &[], fixed).unwrap();
        assert!(!ose.to_string().is_empty());
        assert!(!ose.to_xml().is_empty());
        assert!(!ose.to_json().unwrap().is_empty());
    }
}
