use crate::result::{DmlibResult, Result};
use caith::{Roller, SingleRollResult};

pub fn roll(expr: &str) -> Result {
    let roller = Roller::new(&expr)?;
    let result = roller.roll()?;

    let reason = result.get_reason().map(ToString::to_string);
    let result = match result.get_result() {
        caith::RollResultType::Single(single) => {
            DmlibResult::SingleDiceRoll(single.get_total(), reason)
        }
        caith::RollResultType::Repeated(multi) => DmlibResult::MultiDiceRoll(
            multi.iter().map(SingleRollResult::get_total).collect(),
            reason,
        ),
    };

    Ok(result)
}
