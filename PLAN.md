# dmcli stat block feature — implementation plan

<!-- PROGRESS: current_step=2 -->

## How to use this plan

This file is the single source of truth for feature progress. When invoked
with `agy -p 'implement the next step in the plan'`, an agent should:

1. Read this file from the repo root (`PLAN.md`).
2. Find the step whose status is `[ ]` and whose number matches `current_step`
   in the HTML comment above.
3. Read the step's full specification below.
4. Check out a fresh branch from `main` using the branch name in the step.
5. Implement everything described. Follow `AGENTS.md` at all times.
6. Run the validation commands listed in the step. Fix all errors and warnings
   before proceeding.
7. Commit all changes with the message format: `feat: <step title>`.
8. Update `PLAN.md` **in the same branch**:
   - Mark the current step `[x]`.
   - Increment `current_step` by 1 in the HTML comment.
   - Commit this bookkeeping change as `chore: advance plan to step N`.
9. Push the branch directly using git:
   ```
   git push -u origin <branch>
   ```
10. Stop. Do not implement the next step in the same branch.

> The final step (Step 6) deletes this file and the brain-directory copy.
> After that commit there is no plan to advance.

---

## Overview

Adds a `dmcli statblock` subcommand that generates deterministic, one-line
table-ready stat blocks for use at the DM's table. Supports 5e (default),
Shadowdark, and OSE. Designed to be called by an AI agent that handles flavour
and narrative; the CLI handles all mathematics.

Baseline 5e stats come verbatim from the **Lazy GM's 5e Monster Builder
Resource Document** by Teos Abadía, Scott Fitzgerald Gray, and Michael E. Shea
(CC BY 4.0 — attribution embedded in source).

---

## Output format reference

### 5e (default)
```
Goblin  AC 11  HP 13 (3d6)  Spd 30 ft  CR 1/4
  STR 8(-1) DEX 14(+2) CON 10(+0) INT 10(+0) WIS 8(-1) CHA 8(-1)
  1× Atk +3 to hit, 1d6+2 | Features: Knockdown
```

### Shadowdark
```
Goblin  AC 13  HP 4  Atk dagger +1 (1d4) | MV near | AL Chaos | LVL 1
```

### OSE
```
Goblin  AC 6[13]  HD 1-1  hp 4  Atk 1×weapon (1d6)  MV 60'(20')  ML 7  AL C
```

Each format fits in 80 columns. The 5e block uses `\n  ` continuation indent
for the second and third lines.

---

## Step checklist

- [x] Step 1 — Ability score primitives (`lib::stats`)
- [ ] Step 2 — 5e stat block type (`lib::statblock::StatBlock5e`)
- [ ] Step 3 — LGMRD data module + Shadowdark/OSE types
- [ ] Step 4 — Stat block generation (`lib::statblock::generate`)
- [ ] Step 5 — CLI subcommand (`dmcli statblock`)
- [ ] Step 6 — Cleanup: delete this plan

---

## Steps

---

### Step 1 — Ability score primitives

**Branch:** `feature/statblock-ability-scores`
**Commit message:** `feat: add ability score primitives (lib::stats)`

**Goal:** Add `lib::stats` with foundational ability-score types shared by all
three game systems. No CLI changes in this step.

**Files to create:** `lib/src/stats.rs`
**Files to modify:** `lib/src/lib.rs` — add `pub mod stats;`

#### Implementation

```rust
/// A single ability score, clamped to [1, 30].
pub struct AbilityScore(i8);

impl AbilityScore {
    /// Construct an `AbilityScore`, saturating at the [1, 30] boundary.
    pub fn new(val: i8) -> Self { ... }

    /// Standard 5e modifier: floor((score - 10) / 2).
    pub fn modifier(&self) -> i8 { ... }
}

impl fmt::Display for AbilityScore {
    // Renders as "14(+2)" or "8(-1)". Modifier sign is always shown.
}

/// Six ability scores for a creature.
pub struct AbilityScores {
    pub str: AbilityScore,
    pub dex: AbilityScore,
    pub con: AbilityScore,
    pub int: AbilityScore,
    pub wis: AbilityScore,
    pub cha: AbilityScore,
}

impl fmt::Display for AbilityScores {
    // Renders as:
    // "STR 8(-1) DEX 14(+2) CON 10(+0) INT 10(+0) WIS 8(-1) CHA 8(-1)"
}
```

#### Unit tests (required)

| Input score | Expected modifier |
|---|---|
| 1 | -5 |
| 9 | -1 |
| 10 | 0 |
| 11 | 0 |
| 20 | +5 |
| 30 | +10 |

- Clamp: `AbilityScore::new(0)` → `1`; `AbilityScore::new(31)` → `30`.
- Display: `AbilityScore::new(14).to_string()` == `"14(+2)"`.
- Display: `AbilityScore::new(8).to_string()` == `"8(-1)"`.
- `AbilityScores::to_string()` contains all six labels (`STR`, `DEX`, etc.).

**Validation:**
```
cargo test --workspace
cargo fmt --all
cargo clippy --all-targets --all-features
```
All must pass with zero warnings.

---

### Step 2 — 5e stat block type

**Branch:** `feature/statblock-5e-type`
**Commit message:** `feat: add 5e stat block data type (lib::statblock)`

**Goal:** Implement the 5e stat block data type and all three output modes
(Display/XML/JSON). No generation logic and no CLI changes in this step.

**Prerequisite:** Step 1 merged to `main`. Check out from `main`.

**Files to create:**
- `lib/src/statblock/mod.rs`
- `lib/src/statblock/block_5e.rs`

**Files to modify:** `lib/src/lib.rs` — add `pub mod statblock;`

#### `block_5e.rs`

```rust
use crate::stats::AbilityScores;
use crate::result::Result;

pub struct StatBlock5e {
    pub name: String,
    pub cr: String,              // "1/4", "5", etc.
    pub ac: u8,
    pub hp: u16,
    pub hp_dice: String,         // "3d6", "7d8+14", etc.
    pub speed: u8,               // feet; default 30
    pub scores: AbilityScores,
    pub attacks: Vec<Attack>,
    pub features: Vec<String>,   // feature names, printed on line 3
    pub cr_note: Option<String>, // e.g. "Brute"
}

pub struct Attack {
    pub bonus: i8,
    pub damage_dice: String,     // "1d6+2"
}
```

**`impl Display for StatBlock5e`** — three-line format:
```
{name}  AC {ac}  HP {hp} ({hp_dice})  Spd {speed} ft  CR {cr}
  {scores}
  {n}× Atk {bonus} to hit, {damage_dice}[ | Features: f1, f2, …]
```
- Attack bonus rendered with sign: `+3`, `-1`.
- Features line omitted entirely if `features` is empty.
- Each line fits 80 columns; long feature lists may wrap with `\n    ` indent.

**`StatBlock5e::to_xml(&self) -> String`**
```xml
<statblock system="5e">
  <name>Goblin</name>
  <cr>1/4</cr>
  <ac>11</ac>
  <hp avg="13" dice="3d6"/>
  <speed>30</speed>
  <scores str="8" dex="14" con="10" int="10" wis="8" cha="8"/>
  <attacks count="1" bonus="+3" damage="1d6+2"/>
  <features>Knockdown</features>  <!-- omitted if empty -->
</statblock>
```

**`StatBlock5e::to_json(&self) -> Result<String>`** — using `serde_json::json!`.
Keys: `system`, `name`, `cr`, `ac`, `hp`, `hp_dice`, `speed`, `scores`
(object), `attacks` (array of `{bonus, damage_dice}`), `features` (array,
omitted if empty).

#### Unit tests (required)

- Display contains name, `AC`, `HP`, `Spd`, `CR`.
- Display attack line contains `Atk` and `×`.
- Features line present when features non-empty; absent when empty.
- `to_xml()` starts with `<statblock` and ends with `</statblock>`.
- `to_json()` parses as valid JSON with correct `system` field.

**Validation:**
```
cargo test --workspace
cargo fmt --all
cargo clippy --all-targets --all-features
```

---

### Step 3 — LGMRD data module + Shadowdark and OSE types

**Branch:** `feature/statblock-sd-ose-types`
**Commit message:** `feat: add LGMRD data, Shadowdark and OSE stat block types`

**Goal:** Two deliverables in one branch:

1. `lib/src/statblock/lgmrd.rs` — all LGMRD const data with CC BY 4.0
   attribution comment.
2. `lib/src/statblock/block_sd.rs` and `lib/src/statblock/block_ose.rs` —
   Shadowdark and OSE stat block types.

No CLI changes in this step.

**Prerequisite:** Step 2 merged. Check out from `main`.

**Files to create:**
- `lib/src/statblock/lgmrd.rs`
- `lib/src/statblock/block_sd.rs`
- `lib/src/statblock/block_ose.rs`

**Files to modify:** `lib/src/statblock/mod.rs` — add submodule declarations.

---

#### `lgmrd.rs` — attribution header (required, verbatim)

```rust
// This work includes material taken from the Lazy GM's 5e Monster Builder
// Resource Document written by Teos Abadía of Alphastream.org, Scott
// Fitzgerald Gray of Insaneangel.com, and Michael E. Shea of SlyFlourish.com,
// available under a Creative Commons Attribution 4.0 International License:
// http://creativecommons.org/licenses/by/4.0/
```

#### `lgmrd.rs` — data structures

```rust
pub struct CrRow {
    pub cr: &'static str,
    pub ac: u8,
    pub hp_avg: u16,
    pub hp_dice: &'static str,
    pub prof_bonus: i8,
    pub num_attacks: u8,
    pub damage_dice: &'static str,  // per attack
    pub damage_per_round: u16,
}

pub struct Archetype {
    pub name: &'static str,
    pub cr: &'static str,
    pub str: u8, pub dex: u8, pub con: u8,
    pub int: u8, pub wis: u8, pub cha: u8,
}

pub struct MonsterFeature {
    pub name: &'static str,
    /// CLI-friendly slug, e.g. "damaging-blast"
    pub slug: &'static str,
    pub description: &'static str,
    /// If true, attaching this feature costs 1 attack (minimum 1 remains).
    pub costs_attack: bool,
}

pub struct TacticalRole {
    pub name: &'static str,
    pub summary: &'static str,
}
```

#### `lgmrd.rs` — `CR_TABLE` (34 rows, embed verbatim)

| CR | ac | hp_avg | hp_dice | prof_bonus | num_attacks | damage_dice | damage_per_round |
|---|---|---|---|---|---|---|---|
| "0" | 10 | 3 | "1d4-1" | 2 | 1 | "1d4" | 2 |
| "1/8" | 11 | 9 | "2d8" | 3 | 1 | "1d6+1" | 3 |
| "1/4" | 11 | 13 | "3d6" | 3 | 1 | "1d6+2" | 5 |
| "1/2" | 12 | 22 | "4d8+4" | 4 | 2 | "1d4+2" | 8 |
| "1" | 12 | 33 | "5d8+10" | 5 | 2 | "1d8+2" | 12 |
| "2" | 13 | 45 | "7d8+14" | 5 | 2 | "2d6+2" | 17 |
| "3" | 13 | 65 | "9d8+18" | 5 | 2 | "2d8+3" | 23 |
| "4" | 14 | 84 | "13d8+13" | 6 | 2 | "3d8+1" | 28 |
| "5" | 15 | 95 | "11d8+33" | 7 | 3 | "3d6+2" | 35 |
| "6" | 15 | 112 | "15d8+30" | 7 | 3 | "3d6+4" | 41 |
| "7" | 15 | 130 | "20d8+40" | 7 | 3 | "3d8+3" | 47 |
| "8" | 15 | 136 | "20d8+40" | 7 | 3 | "3d10+2" | 53 |
| "9" | 16 | 145 | "20d8+60" | 8 | 3 | "3d10+3" | 59 |
| "10" | 17 | 155 | "20d10+50" | 9 | 4 | "3d8+3" | 65 |
| "11" | 17 | 165 | "22d8+66" | 9 | 4 | "3d10+2" | 71 |
| "12" | 17 | 175 | "22d8+88" | 9 | 4 | "3d10+3" | 77 |
| "13" | 18 | 184 | "23d8+92" | 10 | 4 | "4d8+3" | 83 |
| "14" | 19 | 196 | "22d10+66" | 11 | 4 | "4d10" | 89 |
| "15" | 19 | 210 | "25d10+75" | 11 | 5 | "3d10+3" | 95 |
| "16" | 19 | 229 | "27d10+81" | 11 | 5 | "4d8+3" | 101 |
| "17" | 20 | 246 | "29d10+87" | 12 | 5 | "3d12+3" | 107 |
| "18" | 21 | 266 | "25d12+100" | 13 | 5 | "4d10+1" | 113 |
| "19" | 21 | 285 | "30d10+100" | 13 | 5 | "4d10+2" | 119 |
| "20" | 21 | 300 | "25d12+125" | 13 | 5 | "4d12" | 132 |
| "21" | 22 | 325 | "27d12+135" | 14 | 5 | "4d12+4" | 150 |
| "22" | 23 | 350 | "29d12+145" | 15 | 5 | "4d12+8" | 168 |
| "23" | 23 | 375 | "31d12+155" | 15 | 5 | "6d10+4" | 186 |
| "24" | 23 | 400 | "33d12+165" | 15 | 5 | "6d10+8" | 204 |
| "25" | 24 | 430 | "35d12+175" | 16 | 5 | "6d10+11" | 222 |
| "26" | 25 | 460 | "37d12+185" | 17 | 5 | "6d10+15" | 240 |
| "27" | 25 | 490 | "39d12+195" | 17 | 5 | "6d10+19" | 258 |
| "28" | 25 | 540 | "43d12+215" | 17 | 5 | "6d10+22" | 276 |
| "29" | 26 | 600 | "47d12+235" | 18 | 5 | "6d10+26" | 294 |
| "30" | 27 | 666 | "53d12+265" | 19 | 5 | "6d10+29" | 312 |

#### `lgmrd.rs` — `ARCHETYPES` (7 entries, embed verbatim)

| name | cr | str | dex | con | int | wis | cha |
|---|---|---|---|---|---|---|---|
| "Minion" | "1/8" | 10 | 12 | 10 | 10 | 12 | 10 |
| "Soldier" | "1/2" | 14 | 12 | 12 | 10 | 10 | 10 |
| "Brute" | "2" | 16 | 12 | 14 | 10 | 10 | 8 |
| "Specialist" | "4" | 12 | 18 | 14 | 10 | 14 | 12 |
| "Myrmidon" | "7" | 10 | 14 | 14 | 18 | 14 | 10 |
| "Sentinel" | "11" | 20 | 16 | 16 | 10 | 14 | 10 |
| "Champion" | "15" | 10 | 12 | 18 | 12 | 16 | 22 |

#### `lgmrd.rs` — `FEATURES` (10 entries, embed verbatim)

| name | slug | costs_attack | description (one sentence for DM use) |
|---|---|---|---|
| "Damaging Blast" | "damaging-blast" | false | "Single-target ranged attack using the standard attack bonus and damage." |
| "Damage Reflection" | "damage-reflection" | true | "Melee attacker takes damage equal to half one of this creature's attacks; costs 1 attack." |
| "Misty Step" | "misty-step" | false | "Bonus action: teleport up to 30 ft to a visible unoccupied space." |
| "Knockdown" | "knockdown" | false | "On hit: target must succeed on a STR save (DC = prof + 8) or be knocked prone." |
| "Restraining Grab" | "restraining-grab" | false | "On hit: target is grappled and restrained; escape DC based on STR or DEX modifier." |
| "Damaging Burst" | "damaging-burst" | false | "Action: 10-ft burst anywhere within 120 ft; DEX/CON/WIS save, half DPR on fail, half on success." |
| "Cunning Action" | "cunning-action" | false | "Bonus action: Dash, Disengage, or Hide." |
| "Damaging Aura" | "damaging-aura" | true | "Creatures starting their turn within 10 ft take half one attack's damage; costs 1 attack." |
| "Energy Weapons" | "energy-weapons" | false | "Weapon attacks deal additional damage of an elemental or magical type." |
| "Damage Transference" | "damage-transference" | false | "On taking damage: transfer half or all to a willing creature within 60 ft." |

#### `lgmrd.rs` — `TACTICAL_ROLES` (6 entries)

| name | summary |
|---|---|
| "Ambusher" | "Uses stealth or surprise to attack before targets can react." |
| "Artillery" | "Deals high damage from range while avoiding melee engagement." |
| "Controller" | "Impairs, restrains, or repositions characters to disrupt their plans." |
| "Defender" | "Absorbs damage and locks characters in place to protect other creatures." |
| "Leader" | "Heals, buffs, or repositions allies to amplify the whole group." |
| "Skirmisher" | "Uses high mobility to dart in for attacks and avoid being pinned down." |

#### `block_sd.rs`

```rust
pub enum SdAlignment { Law, Neutral, Chaos }

pub struct SdAttack {
    pub name: String,     // e.g. "dagger"
    pub bonus: i8,
    pub dice: String,     // e.g. "1d4"
}

pub struct StatBlockSd {
    pub name: String,
    pub ac: u8,
    pub hp: u8,
    pub attacks: Vec<SdAttack>,
    pub mv: String,            // "near" | "close" | "far"
    pub alignment: SdAlignment,
    pub level: u8,
}
```

`impl Display for StatBlockSd` → one line:
```
{name}  AC {ac}  HP {hp}  Atk {name} {bonus} ({dice})[, …] | MV {mv} | AL {al} | LVL {level}
```

`StatBlockSd::to_xml`, `StatBlockSd::to_json` per the same pattern as
`StatBlock5e`.

#### `block_ose.rs`

```rust
pub enum OseAlignment { Lawful, Neutral, Chaotic }

pub struct OseAttack {
    pub count: u8,        // e.g. 1
    pub weapon: String,   // e.g. "weapon"
    pub dice: String,     // e.g. "1d6"
}

pub struct StatBlockOse {
    pub name: String,
    pub ac_descending: u8,
    pub ac_ascending: u8,
    pub hd: String,        // "1-1", "2", "3+1"
    pub hp: u8,
    pub attacks: Vec<OseAttack>,
    pub mv_ft: u16,
    pub mv_ft_enc: u16,
    pub morale: u8,
    pub alignment: OseAlignment,
}
```

`impl Display for StatBlockOse` → one line:
```
{name}  AC {asc}[{desc}]  HD {hd}  hp {hp}  Atk {n}×{weapon} ({dice})[, …]  MV {mv}'({enc}')  ML {ml}  AL {al}
```

`StatBlockOse::to_xml`, `StatBlockOse::to_json` per the same pattern.

#### Unit tests (required)

- `lgmrd::CR_TABLE` has 34 entries.
- `lgmrd::CR_TABLE[0].cr == "0"` and `lgmrd::CR_TABLE[33].cr == "30"`.
- `lgmrd::ARCHETYPES` has 7 entries.
- `lgmrd::FEATURES` has 10 entries.
- Exactly 2 features have `costs_attack == true`.
- `lgmrd::TACTICAL_ROLES` has 6 entries.
- `StatBlockSd` display contains `AC`, `HP`, `Atk`, `MV`, `AL`, `LVL`.
- `StatBlockOse` display contains `AC`, `HD`, `hp`, `Atk`, `MV`, `ML`, `AL`.

**Validation:**
```
cargo test --workspace
cargo fmt --all
cargo clippy --all-targets --all-features
```

---

### Step 4 — Stat block generation

**Branch:** `feature/statblock-generate`
**Commit message:** `feat: add stat block generation (lib::statblock::generate)`

**Goal:** Implement `lib::statblock::generate`, which looks up the LGMRD CR
table, picks an archetype, optionally attaches features, rolls HP live, and
returns a fully-populated `GeneratedStatBlock`. Shadowdark and OSE generation
uses simple formulas.

**Prerequisite:** Step 3 merged. Check out from `main`.

**Files to create:** `lib/src/statblock/generate.rs`
**Files to modify:** `lib/src/statblock/mod.rs` — expose public types and
`generate` function.

#### Public API

```rust
// in lib/src/statblock/mod.rs

pub use block_5e::{Attack, StatBlock5e};
pub use block_sd::{SdAlignment, SdAttack, StatBlockSd};
pub use block_ose::{OseAlignment, OseAttack, StatBlockOse};
pub use lgmrd::{Archetype, CrRow, MonsterFeature, TacticalRole,
                ARCHETYPES, CR_TABLE, FEATURES, TACTICAL_ROLES};

pub enum System { Dnd5e, Shadowdark, Ose }

/// Combat archetype — determines ability score array for 5e generation.
/// Defaults to `Brute` when not specified.
pub enum Role {
    Minion, Soldier, Brute, Specialist, Myrmidon, Sentinel, Champion,
}

pub enum GeneratedStatBlock {
    Dnd5e(StatBlock5e),
    Shadowdark(StatBlockSd),
    Ose(StatBlockOse),
}

impl fmt::Display for GeneratedStatBlock { /* delegate */ }
impl GeneratedStatBlock {
    pub fn to_xml(&self) -> String { /* delegate */ }
    pub fn to_json(&self) -> crate::Result<String> { /* delegate */ }
}

/// Generate a stat block.
///
/// `features` is a slice of slugs (e.g. `"knockdown"`) matched
/// case-insensitively against `lgmrd::FEATURES`. Unknown slugs return
/// `Err`. Features with `costs_attack = true` reduce attack count by 1,
/// with a floor of 1.
///
/// HP is rolled using `roll_fn`, which accepts a dice expression string
/// and returns `i64`. In production pass `|expr| dice::roll(expr).map(|r| r.total())`.
/// In tests pass a closure returning a fixed value.
pub fn generate(
    name: &str,
    cr_or_level: &str,
    system: System,
    role: Option<Role>,
    features: &[&str],
    roll_fn: impl Fn(&str) -> crate::Result<i64>,
) -> crate::Result<GeneratedStatBlock>;
```

#### Generation rules

**5e:**
1. Look up `cr_or_level` in `CR_TABLE` (exact string match). Return
   `Err` for unknown CRs.
2. Select archetype: use `role` if given, else default to `Brute`.
   Override `hp_dice` and `num_attacks` from `CR_TABLE` (archetype only
   provides ability scores and narrative name).
3. Roll HP: `roll_fn(cr_row.hp_dice)?` clamped to `[1, u16::MAX]`.
4. Resolve features: for each slug, find the matching `MonsterFeature`.
   Accumulate `costs_attack` count; subtract from `num_attacks`, floor 1.
5. Build `StatBlock5e` with `speed = 30`, `cr_note = Some(archetype.name)`,
   `features` = feature names, single `Attack` with `bonus = prof_bonus`
   and `damage_dice = cr_row.damage_dice`.

**Shadowdark:**
- Parse `cr_or_level` as `u8` level (1–20). Return `Err` otherwise.
- Roll HP: `roll_fn(&format!("{}d6", level))?`.
- AC = 10 (no DEX modifier without full archetype — keep it simple).
- One attack: `+level to hit`, `1d6` damage.
- `mv = "near"`, `alignment = Neutral`.
- Features list ignored (SD format has no feature line).

**OSE:**
- Parse `cr_or_level` as HD string: `"N"`, `"N-M"`, or `"N+M"`.
  Extract the base die count `N` as a `u8`.
- Roll HP: `roll_fn(&format!("{}d8", n))?`.
- `ac_ascending = 10 + n`, `ac_descending = 19 - ac_ascending` (simple
  approximation; DM adjusts at table).
- One attack: `1×weapon`, `1d6`.
- `mv_ft = 60`, `mv_ft_enc = 20`, `morale = 7`, `alignment = Neutral`.
- Features list ignored (OSE format has no feature line).

#### `DiceRoll` total helper

Add `DiceRoll::total(&self) -> i64` in `lib/src/dice.rs`:
- `Single(v, _)` → `v`
- `Multi(vs, _)` → `vs.iter().sum()`

This keeps the generate closure clean: `|expr| dice::roll(expr).map(|r| r.total())`.

#### Unit tests (required)

Use a fixed roll closure `|_| Ok(10_i64)` for all HP tests:

- `generate("Goblin", "1/4", System::Dnd5e, None, &[], fixed)` → HP = 10,
  AC = 11, attack bonus = +3, 1 attack, no features.
- `generate` with `features = &["knockdown"]` → features list = `["Knockdown"]`,
  attack count unchanged.
- `generate` with `features = &["damage-reflection"]` → attack count reduced by 1.
- Two `costs_attack` features on a 1-attack creature → still 1 attack.
- Unknown feature slug → `Err`.
- Unknown CR → `Err`.
- Shadowdark level `"3"` → HP = 10 (from fixed closure), AC = 10, level = 3.
- OSE HD `"2"` → HP = 10, ac_ascending = 12, `hd = "2"`.
- OSE HD `"1-1"` → base die count = 1.
- `GeneratedStatBlock::to_string()`, `to_xml()`, `to_json()` all non-empty.

**Validation:**
```
cargo test --workspace
cargo fmt --all
cargo clippy --all-targets --all-features
```

---

### Step 5 — CLI subcommand

**Branch:** `feature/statblock-cli`
**Commit message:** `feat: add dmcli statblock subcommand`

**Goal:** Expose `lib::statblock::generate` as `dmcli statblock`, wiring up
all options. No library changes.

**Prerequisite:** Step 4 merged. Check out from `main`.

**Files to modify:** `cli/src/main.rs`

#### Command shape

```
dmcli statblock [OPTIONS] <NAME> <CR_OR_LEVEL>

Arguments:
  <NAME>          Monster name (quote multi-word names)
  <CR_OR_LEVEL>   CR for 5e ("0", "1/8", "1/4", "1/2", "1"–"30"),
                  level for Shadowdark ("1"–"20"),
                  HD string for OSE ("2", "1-1", "3+1")

Options:
  -s, --system <SYSTEM>    Rule system [default: 5e]
                           [possible values: 5e, shadowdark, ose]
  -r, --role <ROLE>        Ability score archetype for 5e [default: brute]
                           [possible values: minion, soldier, brute, specialist,
                            myrmidon, sentinel, champion]
  -f, --feature <FEATURE>  Monster feature slug, repeatable [default: none]
                           [possible values: damaging-blast, damage-reflection,
                            misty-step, knockdown, restraining-grab,
                            damaging-burst, cunning-action, damaging-aura,
                            energy-weapons, damage-transference]
  -o, --output <FORMAT>    Output format [default: markdown]
                           [possible values: markdown, xml, json]
  -h, --help               Print help
```

#### `CliCommand::Statblock` variant

```rust
Statblock {
    #[arg(short, long, default_value = "5e")]
    system: StatblockSystem,

    #[arg(short, long, default_value = "brute")]
    role: StatblockRole,

    #[arg(short, long)]
    feature: Vec<String>,

    #[arg(short, long, default_value = "markdown")]
    output: OutputFormat,

    name: String,
    cr_or_level: String,
}
```

Add `StatblockSystem` and `StatblockRole` as `ValueEnum` types in `main.rs`,
mapping to `lib::statblock::System` and `lib::statblock::Role`.

The `generate` call uses the real roll closure:
```rust
|expr| {
    dice::roll(expr).map(|r| r.total())
}
```

Print result using existing `OutputFormat` match pattern.

#### Smoke tests (run manually)

```
dmcli statblock "Goblin" "1/4"
dmcli statblock "Shadow Knight" "2" --role brute --feature knockdown --feature misty-step
dmcli statblock "Wraith" "5" --role myrmidon --output json
dmcli statblock --system shadowdark "Cave Troll" "3"
dmcli statblock --system ose "Orc" "1" --output xml
dmcli statblock --help
```

**Validation:**
```
cargo test --workspace
cargo fmt --all
cargo clippy --all-targets --all-features
```

---

### Step 6 — Cleanup

**Branch:** `feature/statblock-cleanup`
**Commit message:** `chore: remove implementation plan`

**Goal:** Delete this plan file. Confirm the full feature is green.

**Prerequisite:** Step 5 merged. Check out from `main`.

**Actions:**
1. `git rm PLAN.md`
2. Delete brain-directory copy (path is below — only if it exists):
   `/home/matt/.gemini/antigravity-cli/brain/8e7e0243-6a80-4f62-ba45-7c409ac1ecb4/plan.md`
3. Run the full validation suite:
   ```
   cargo test --workspace
   cargo fmt --all
   cargo clippy --all-targets --all-features
   dmcli statblock --help
   ```
4. Commit as `chore: remove implementation plan`.
5. Push the branch directly using `git push -u origin <branch>`. No bookkeeping update needed — the file is gone.

---

## Module map (final state after Step 5)

```
lib/src/
  lib.rs                — pub mod dice; pub mod stats; pub mod statblock;
  dice.rs               — existing + DiceRoll::total()
  error.rs              — existing
  result.rs             — existing
  stats.rs              — AbilityScore, AbilityScores  [Step 1]
  statblock/
    mod.rs              — System, Role, GeneratedStatBlock, generate()
    block_5e.rs         — StatBlock5e, Attack  [Step 2]
    block_sd.rs         — StatBlockSd, SdAttack, SdAlignment  [Step 3]
    block_ose.rs        — StatBlockOse, OseAttack, OseAlignment  [Step 3]
    lgmrd.rs            — CrRow/Archetype/MonsterFeature/TacticalRole + consts  [Step 3]
    generate.rs         — generate()  [Step 4]

cli/src/
  main.rs               — existing + CliCommand::Statblock  [Step 5]
  result.rs             — existing
```
