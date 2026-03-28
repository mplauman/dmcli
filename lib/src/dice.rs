use crate::result::Result;
use caith::{Roller, SingleRollResult};

pub fn roll(expr: &str) -> Result<DiceRoll> {
    let roller = Roller::new(expr)?;
    let result = roller.roll()?;

    let reason = result.get_reason().map(ToString::to_string);
    let result = match result.get_result() {
        caith::RollResultType::Single(single) => DiceRoll::Single(single.get_total(), reason),
        caith::RollResultType::Repeated(multi) => DiceRoll::Multi(
            multi.iter().map(SingleRollResult::get_total).collect(),
            reason,
        ),
    };

    Ok(result)
}

pub enum DiceRoll {
    Single(i64, Option<String>),
    Multi(Vec<i64>, Option<String>),
}
