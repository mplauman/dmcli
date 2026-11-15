//! Stat block generation and data structures for D&D 5e, Shadowdark, and OSE.

pub mod block_5e;
pub mod block_ose;
pub mod block_sd;
pub mod lgmrd;

pub use block_5e::{Attack, StatBlock5e};
pub use block_ose::{OseAlignment, OseAttack, StatBlockOse};
pub use block_sd::{SdAlignment, SdAttack, StatBlockSd};
pub use lgmrd::{
    ARCHETYPES, Archetype, CR_TABLE, CrRow, FEATURES, MonsterFeature, TACTICAL_ROLES, TacticalRole,
};
