//! Ability score primitives for D&D 5e, Shadowdark, and OSE.

use std::fmt;

/// A single ability score, clamped to the range [1, 30].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbilityScore(i8);

impl AbilityScore {
    /// Construct an `AbilityScore`, saturating at the [1, 30] boundary.
    pub fn new(val: i8) -> Self {
        Self(val.clamp(1, 30))
    }

    /// Standard 5e modifier: floor((score - 10) / 2).
    pub fn modifier(&self) -> i8 {
        (self.0 - 10).div_euclid(2)
    }

    /// Return the raw score value, in the range [1, 30].
    pub fn score(&self) -> i8 {
        self.0
    }
}

impl fmt::Display for AbilityScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({:+})", self.0, self.modifier())
    }
}

/// Six ability scores for a creature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbilityScores {
    /// Strength score.
    pub str: AbilityScore,
    /// Dexterity score.
    pub dex: AbilityScore,
    /// Constitution score.
    pub con: AbilityScore,
    /// Intelligence score.
    pub int: AbilityScore,
    /// Wisdom score.
    pub wis: AbilityScore,
    /// Charisma score.
    pub cha: AbilityScore,
}

impl fmt::Display for AbilityScores {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "STR {} DEX {} CON {} INT {} WIS {} CHA {}",
            self.str, self.dex, self.con, self.int, self.wis, self.cha
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modifier_table() {
        let cases = [(1, -5), (9, -1), (10, 0), (11, 0), (20, 5), (30, 10)];

        for (score, expected_mod) in cases {
            assert_eq!(
                AbilityScore::new(score).modifier(),
                expected_mod,
                "score {score} should have modifier {expected_mod}"
            );
        }
    }

    #[test]
    fn test_clamping() {
        assert_eq!(AbilityScore::new(0).score(), 1);
        assert_eq!(AbilityScore::new(-5).score(), 1);
        assert_eq!(AbilityScore::new(31).score(), 30);
        assert_eq!(AbilityScore::new(50).score(), 30);
        assert_eq!(AbilityScore::new(15).score(), 15);
    }

    #[test]
    fn test_display_ability_score() {
        assert_eq!(AbilityScore::new(14).to_string(), "14(+2)");
        assert_eq!(AbilityScore::new(8).to_string(), "8(-1)");
        assert_eq!(AbilityScore::new(10).to_string(), "10(+0)");
        assert_eq!(AbilityScore::new(1).to_string(), "1(-5)");
        assert_eq!(AbilityScore::new(30).to_string(), "30(+10)");
    }

    #[test]
    fn test_display_ability_scores() {
        let scores = AbilityScores {
            str: AbilityScore::new(8),
            dex: AbilityScore::new(14),
            con: AbilityScore::new(10),
            int: AbilityScore::new(10),
            wis: AbilityScore::new(8),
            cha: AbilityScore::new(8),
        };

        let rendered = scores.to_string();
        assert_eq!(
            rendered,
            "STR 8(-1) DEX 14(+2) CON 10(+0) INT 10(+0) WIS 8(-1) CHA 8(-1)"
        );

        for label in ["STR", "DEX", "CON", "INT", "WIS", "CHA"] {
            assert!(
                rendered.contains(label),
                "rendered scores should contain label {label}"
            );
        }
    }
}
