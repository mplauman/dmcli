// This work includes material taken from the Lazy GM's 5e Monster Builder
// Resource Document written by Teos Abadía of Alphastream.org, Scott
// Fitzgerald Gray of Insaneangel.com, and Michael E. Shea of SlyFlourish.com,
// available under a Creative Commons Attribution 4.0 International License:
// http://creativecommons.org/licenses/by/4.0/

//! Constant baseline tables and archetype definitions from the Lazy GM's 5e Monster Builder Resource Document.

/// A row in the LGMRD Challenge Rating baseline table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CrRow {
    /// Challenge rating string (e.g., "0", "1/4", "30").
    pub cr: &'static str,
    /// Baseline armor class.
    pub ac: u8,
    /// Baseline hit point average.
    pub hp_avg: u16,
    /// Baseline hit point dice formula.
    pub hp_dice: &'static str,
    /// Proficiency bonus.
    pub prof_bonus: i8,
    /// Number of attacks.
    pub num_attacks: u8,
    /// Damage dice expression per attack.
    pub damage_dice: &'static str,
    /// Damage per round total across all attacks.
    pub damage_per_round: u16,
}

/// A combat archetype providing ability scores and thematic role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Archetype {
    /// Archetype name (e.g., "Brute", "Soldier").
    pub name: &'static str,
    /// Baseline challenge rating.
    pub cr: &'static str,
    /// Strength score.
    pub str: u8,
    /// Dexterity score.
    pub dex: u8,
    /// Constitution score.
    pub con: u8,
    /// Intelligence score.
    pub int: u8,
    /// Wisdom score.
    pub wis: u8,
    /// Charisma score.
    pub cha: u8,
}

/// A special monster feature that can be attached to a 5e stat block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MonsterFeature {
    /// Feature name.
    pub name: &'static str,
    /// CLI-friendly slug, e.g. "damaging-blast".
    pub slug: &'static str,
    /// One sentence description for DM use.
    pub description: &'static str,
    /// If true, attaching this feature costs 1 attack (minimum 1 remains).
    pub costs_attack: bool,
}

/// A tactical role describing how a creature behaves in combat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TacticalRole {
    /// Role name (e.g., "Ambusher", "Artillery").
    pub name: &'static str,
    /// Tactical summary.
    pub summary: &'static str,
}

/// The 34-row Challenge Rating baseline table from the LGMRD.
pub const CR_TABLE: [CrRow; 34] = [
    CrRow {
        cr: "0",
        ac: 10,
        hp_avg: 3,
        hp_dice: "1d4-1",
        prof_bonus: 2,
        num_attacks: 1,
        damage_dice: "1d4",
        damage_per_round: 2,
    },
    CrRow {
        cr: "1/8",
        ac: 11,
        hp_avg: 9,
        hp_dice: "2d8",
        prof_bonus: 3,
        num_attacks: 1,
        damage_dice: "1d6+1",
        damage_per_round: 3,
    },
    CrRow {
        cr: "1/4",
        ac: 11,
        hp_avg: 13,
        hp_dice: "3d6",
        prof_bonus: 3,
        num_attacks: 1,
        damage_dice: "1d6+2",
        damage_per_round: 5,
    },
    CrRow {
        cr: "1/2",
        ac: 12,
        hp_avg: 22,
        hp_dice: "4d8+4",
        prof_bonus: 4,
        num_attacks: 2,
        damage_dice: "1d4+2",
        damage_per_round: 8,
    },
    CrRow {
        cr: "1",
        ac: 12,
        hp_avg: 33,
        hp_dice: "5d8+10",
        prof_bonus: 5,
        num_attacks: 2,
        damage_dice: "1d8+2",
        damage_per_round: 12,
    },
    CrRow {
        cr: "2",
        ac: 13,
        hp_avg: 45,
        hp_dice: "7d8+14",
        prof_bonus: 5,
        num_attacks: 2,
        damage_dice: "2d6+2",
        damage_per_round: 17,
    },
    CrRow {
        cr: "3",
        ac: 13,
        hp_avg: 65,
        hp_dice: "9d8+18",
        prof_bonus: 5,
        num_attacks: 2,
        damage_dice: "2d8+3",
        damage_per_round: 23,
    },
    CrRow {
        cr: "4",
        ac: 14,
        hp_avg: 84,
        hp_dice: "13d8+13",
        prof_bonus: 6,
        num_attacks: 2,
        damage_dice: "3d8+1",
        damage_per_round: 28,
    },
    CrRow {
        cr: "5",
        ac: 15,
        hp_avg: 95,
        hp_dice: "11d8+33",
        prof_bonus: 7,
        num_attacks: 3,
        damage_dice: "3d6+2",
        damage_per_round: 35,
    },
    CrRow {
        cr: "6",
        ac: 15,
        hp_avg: 112,
        hp_dice: "15d8+30",
        prof_bonus: 7,
        num_attacks: 3,
        damage_dice: "3d6+4",
        damage_per_round: 41,
    },
    CrRow {
        cr: "7",
        ac: 15,
        hp_avg: 130,
        hp_dice: "20d8+40",
        prof_bonus: 7,
        num_attacks: 3,
        damage_dice: "3d8+3",
        damage_per_round: 47,
    },
    CrRow {
        cr: "8",
        ac: 15,
        hp_avg: 136,
        hp_dice: "20d8+40",
        prof_bonus: 7,
        num_attacks: 3,
        damage_dice: "3d10+2",
        damage_per_round: 53,
    },
    CrRow {
        cr: "9",
        ac: 16,
        hp_avg: 145,
        hp_dice: "20d8+60",
        prof_bonus: 8,
        num_attacks: 3,
        damage_dice: "3d10+3",
        damage_per_round: 59,
    },
    CrRow {
        cr: "10",
        ac: 17,
        hp_avg: 155,
        hp_dice: "20d10+50",
        prof_bonus: 9,
        num_attacks: 4,
        damage_dice: "3d8+3",
        damage_per_round: 65,
    },
    CrRow {
        cr: "11",
        ac: 17,
        hp_avg: 165,
        hp_dice: "22d8+66",
        prof_bonus: 9,
        num_attacks: 4,
        damage_dice: "3d10+2",
        damage_per_round: 71,
    },
    CrRow {
        cr: "12",
        ac: 17,
        hp_avg: 175,
        hp_dice: "22d8+88",
        prof_bonus: 9,
        num_attacks: 4,
        damage_dice: "3d10+3",
        damage_per_round: 77,
    },
    CrRow {
        cr: "13",
        ac: 18,
        hp_avg: 184,
        hp_dice: "23d8+92",
        prof_bonus: 10,
        num_attacks: 4,
        damage_dice: "4d8+3",
        damage_per_round: 83,
    },
    CrRow {
        cr: "14",
        ac: 19,
        hp_avg: 196,
        hp_dice: "22d10+66",
        prof_bonus: 11,
        num_attacks: 4,
        damage_dice: "4d10",
        damage_per_round: 89,
    },
    CrRow {
        cr: "15",
        ac: 19,
        hp_avg: 210,
        hp_dice: "25d10+75",
        prof_bonus: 11,
        num_attacks: 5,
        damage_dice: "3d10+3",
        damage_per_round: 95,
    },
    CrRow {
        cr: "16",
        ac: 19,
        hp_avg: 229,
        hp_dice: "27d10+81",
        prof_bonus: 11,
        num_attacks: 5,
        damage_dice: "4d8+3",
        damage_per_round: 101,
    },
    CrRow {
        cr: "17",
        ac: 20,
        hp_avg: 246,
        hp_dice: "29d10+87",
        prof_bonus: 12,
        num_attacks: 5,
        damage_dice: "3d12+3",
        damage_per_round: 107,
    },
    CrRow {
        cr: "18",
        ac: 21,
        hp_avg: 266,
        hp_dice: "25d12+100",
        prof_bonus: 13,
        num_attacks: 5,
        damage_dice: "4d10+1",
        damage_per_round: 113,
    },
    CrRow {
        cr: "19",
        ac: 21,
        hp_avg: 285,
        hp_dice: "30d10+100",
        prof_bonus: 13,
        num_attacks: 5,
        damage_dice: "4d10+2",
        damage_per_round: 119,
    },
    CrRow {
        cr: "20",
        ac: 21,
        hp_avg: 300,
        hp_dice: "25d12+125",
        prof_bonus: 13,
        num_attacks: 5,
        damage_dice: "4d12",
        damage_per_round: 132,
    },
    CrRow {
        cr: "21",
        ac: 22,
        hp_avg: 325,
        hp_dice: "27d12+135",
        prof_bonus: 14,
        num_attacks: 5,
        damage_dice: "4d12+4",
        damage_per_round: 150,
    },
    CrRow {
        cr: "22",
        ac: 23,
        hp_avg: 350,
        hp_dice: "29d12+145",
        prof_bonus: 15,
        num_attacks: 5,
        damage_dice: "4d12+8",
        damage_per_round: 168,
    },
    CrRow {
        cr: "23",
        ac: 23,
        hp_avg: 375,
        hp_dice: "31d12+155",
        prof_bonus: 15,
        num_attacks: 5,
        damage_dice: "6d10+4",
        damage_per_round: 186,
    },
    CrRow {
        cr: "24",
        ac: 23,
        hp_avg: 400,
        hp_dice: "33d12+165",
        prof_bonus: 15,
        num_attacks: 5,
        damage_dice: "6d10+8",
        damage_per_round: 204,
    },
    CrRow {
        cr: "25",
        ac: 24,
        hp_avg: 430,
        hp_dice: "35d12+175",
        prof_bonus: 16,
        num_attacks: 5,
        damage_dice: "6d10+11",
        damage_per_round: 222,
    },
    CrRow {
        cr: "26",
        ac: 25,
        hp_avg: 460,
        hp_dice: "37d12+185",
        prof_bonus: 17,
        num_attacks: 5,
        damage_dice: "6d10+15",
        damage_per_round: 240,
    },
    CrRow {
        cr: "27",
        ac: 25,
        hp_avg: 490,
        hp_dice: "39d12+195",
        prof_bonus: 17,
        num_attacks: 5,
        damage_dice: "6d10+19",
        damage_per_round: 258,
    },
    CrRow {
        cr: "28",
        ac: 25,
        hp_avg: 540,
        hp_dice: "43d12+215",
        prof_bonus: 17,
        num_attacks: 5,
        damage_dice: "6d10+22",
        damage_per_round: 276,
    },
    CrRow {
        cr: "29",
        ac: 26,
        hp_avg: 600,
        hp_dice: "47d12+235",
        prof_bonus: 18,
        num_attacks: 5,
        damage_dice: "6d10+26",
        damage_per_round: 294,
    },
    CrRow {
        cr: "30",
        ac: 27,
        hp_avg: 666,
        hp_dice: "53d12+265",
        prof_bonus: 19,
        num_attacks: 5,
        damage_dice: "6d10+29",
        damage_per_round: 312,
    },
];

/// The 7 combat archetypes from the LGMRD.
pub const ARCHETYPES: [Archetype; 7] = [
    Archetype {
        name: "Minion",
        cr: "1/8",
        str: 10,
        dex: 12,
        con: 10,
        int: 10,
        wis: 12,
        cha: 10,
    },
    Archetype {
        name: "Soldier",
        cr: "1/2",
        str: 14,
        dex: 12,
        con: 12,
        int: 10,
        wis: 10,
        cha: 10,
    },
    Archetype {
        name: "Brute",
        cr: "2",
        str: 16,
        dex: 12,
        con: 14,
        int: 10,
        wis: 10,
        cha: 8,
    },
    Archetype {
        name: "Specialist",
        cr: "4",
        str: 12,
        dex: 18,
        con: 14,
        int: 10,
        wis: 14,
        cha: 12,
    },
    Archetype {
        name: "Myrmidon",
        cr: "7",
        str: 10,
        dex: 14,
        con: 14,
        int: 18,
        wis: 14,
        cha: 10,
    },
    Archetype {
        name: "Sentinel",
        cr: "11",
        str: 20,
        dex: 16,
        con: 16,
        int: 10,
        wis: 14,
        cha: 10,
    },
    Archetype {
        name: "Champion",
        cr: "15",
        str: 10,
        dex: 12,
        con: 18,
        int: 12,
        wis: 16,
        cha: 22,
    },
];

/// The 10 monster features from the LGMRD.
pub const FEATURES: [MonsterFeature; 10] = [
    MonsterFeature {
        name: "Damaging Blast",
        slug: "damaging-blast",
        costs_attack: false,
        description: "Single-target ranged attack using the standard attack bonus and damage.",
    },
    MonsterFeature {
        name: "Damage Reflection",
        slug: "damage-reflection",
        costs_attack: true,
        description: "Melee attacker takes damage equal to half one of this creature's attacks; costs 1 attack.",
    },
    MonsterFeature {
        name: "Misty Step",
        slug: "misty-step",
        costs_attack: false,
        description: "Bonus action: teleport up to 30 ft to a visible unoccupied space.",
    },
    MonsterFeature {
        name: "Knockdown",
        slug: "knockdown",
        costs_attack: false,
        description: "On hit: target must succeed on a STR save (DC = prof + 8) or be knocked prone.",
    },
    MonsterFeature {
        name: "Restraining Grab",
        slug: "restraining-grab",
        costs_attack: false,
        description: "On hit: target is grappled and restrained; escape DC based on STR or DEX modifier.",
    },
    MonsterFeature {
        name: "Damaging Burst",
        slug: "damaging-burst",
        costs_attack: false,
        description: "Action: 10-ft burst anywhere within 120 ft; DEX/CON/WIS save, half DPR on fail, half on success.",
    },
    MonsterFeature {
        name: "Cunning Action",
        slug: "cunning-action",
        costs_attack: false,
        description: "Bonus action: Dash, Disengage, or Hide.",
    },
    MonsterFeature {
        name: "Damaging Aura",
        slug: "damaging-aura",
        costs_attack: true,
        description: "Creatures starting their turn within 10 ft take half one attack's damage; costs 1 attack.",
    },
    MonsterFeature {
        name: "Energy Weapons",
        slug: "energy-weapons",
        costs_attack: false,
        description: "Weapon attacks deal additional damage of an elemental or magical type.",
    },
    MonsterFeature {
        name: "Damage Transference",
        slug: "damage-transference",
        costs_attack: false,
        description: "On taking damage: transfer half or all to a willing creature within 60 ft.",
    },
];

/// The 6 tactical roles from the LGMRD.
pub const TACTICAL_ROLES: [TacticalRole; 6] = [
    TacticalRole {
        name: "Ambusher",
        summary: "Uses stealth or surprise to attack before targets can react.",
    },
    TacticalRole {
        name: "Artillery",
        summary: "Deals high damage from range while avoiding melee engagement.",
    },
    TacticalRole {
        name: "Controller",
        summary: "Impairs, restrains, or repositions characters to disrupt their plans.",
    },
    TacticalRole {
        name: "Defender",
        summary: "Absorbs damage and locks characters in place to protect other creatures.",
    },
    TacticalRole {
        name: "Leader",
        summary: "Heals, buffs, or repositions allies to amplify the whole group.",
    },
    TacticalRole {
        name: "Skirmisher",
        summary: "Uses high mobility to dart in for attacks and avoid being pinned down.",
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cr_table_has_34_entries() {
        assert_eq!(CR_TABLE.len(), 34);
    }

    #[test]
    fn cr_table_boundaries() {
        assert_eq!(CR_TABLE[0].cr, "0");
        assert_eq!(CR_TABLE[33].cr, "30");
    }

    #[test]
    fn archetypes_count() {
        assert_eq!(ARCHETYPES.len(), 7);
    }

    #[test]
    fn features_count() {
        assert_eq!(FEATURES.len(), 10);
    }

    #[test]
    fn features_costs_attack_count() {
        let count = FEATURES.iter().filter(|f| f.costs_attack).count();
        assert_eq!(count, 2);
    }

    #[test]
    fn tactical_roles_count() {
        assert_eq!(TACTICAL_ROLES.len(), 6);
    }
}
