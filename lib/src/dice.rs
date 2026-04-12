use std::fmt;
use std::fmt::Write as FmtWrite;

use crate::result::Result;
use caith::{Roller, SingleRollResult};

/// Roll a dice expression and return the result.
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

/// The result of a dice roll expression.
///
/// Three output modes are available:
///
/// - [`fmt::Display`]: Markdown, intended for human consumption at a terminal
///   or in an editor.
/// - [`DiceRoll::to_xml`]: an XML element suitable for injection into an LLM
///   prompt as context.
/// - [`DiceRoll::to_json`]: a JSON object for structured agent-skill responses.
pub enum DiceRoll {
    /// A single roll result with an optional reason label.
    Single(i64, Option<String>),
    /// Multiple roll results (from a repeated expression) with an optional
    /// reason label.
    Multi(Vec<i64>, Option<String>),
}

impl fmt::Display for DiceRoll {
    /// Renders the roll as a Markdown string for human-readable output.
    ///
    /// - Single: `🎲 **{total}**` with an optional `*({reason})*` suffix.
    /// - Multi: `🎲🎲 {v1}, {v2}, …` with an optional `*({reason})*` and a
    ///   `— total: **{sum}**` suffix.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiceRoll::Single(total, reason) => {
                write!(f, "🎲 **{total}**")?;
                if let Some(r) = reason {
                    write!(f, " *({r})*")?;
                }
                Ok(())
            }
            DiceRoll::Multi(values, reason) => {
                let total: i64 = values.iter().sum();
                let values_str = values
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "🎲🎲 {values_str}")?;
                if let Some(r) = reason {
                    write!(f, " *({r})*")?;
                }
                write!(f, " — total: **{total}**")
            }
        }
    }
}

impl DiceRoll {
    /// Renders the roll as an XML element for injection into an LLM prompt.
    ///
    /// ```text
    /// <roll type="single">
    /// <total>42</total>
    /// <reason>attack roll</reason>   <!-- omitted when absent -->
    /// </roll>
    /// ```
    ///
    /// For multi-rolls, a `<values>` tag lists the individual results as a
    /// comma-separated string before the `<total>`.
    pub fn to_xml(&self) -> String {
        let mut buf = String::new();
        match self {
            DiceRoll::Single(total, reason) => {
                writeln!(buf, "<roll type=\"single\">").expect("write to String is infallible");
                writeln!(buf, "<total>{total}</total>").expect("write to String is infallible");
                if let Some(r) = reason {
                    writeln!(buf, "<reason>{r}</reason>").expect("write to String is infallible");
                }
                write!(buf, "</roll>").expect("write to String is infallible");
            }
            DiceRoll::Multi(values, reason) => {
                let total: i64 = values.iter().sum();
                let values_str = values
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                writeln!(buf, "<roll type=\"multi\">").expect("write to String is infallible");
                writeln!(buf, "<values>{values_str}</values>")
                    .expect("write to String is infallible");
                writeln!(buf, "<total>{total}</total>").expect("write to String is infallible");
                if let Some(r) = reason {
                    writeln!(buf, "<reason>{r}</reason>").expect("write to String is infallible");
                }
                write!(buf, "</roll>").expect("write to String is infallible");
            }
        }
        buf
    }

    /// Serializes the roll to a pretty-printed JSON object for agent-skill
    /// responses.
    ///
    /// Fields:
    ///
    /// | Field    | Type    | Description                                          |
    /// |----------|---------|------------------------------------------------------|
    /// | `type`   | string  | `"single"` or `"multi"`                              |
    /// | `total`  | integer | The roll total (sum of all values for multi-rolls)   |
    /// | `values` | array   | Individual roll values (multi only)                  |
    /// | `reason` | string  | Label from the expression (omitted when not present) |
    pub fn to_json(&self) -> Result<String> {
        let value = match self {
            DiceRoll::Single(total, reason) => {
                let mut map = serde_json::Map::new();
                map.insert("type".to_string(), serde_json::json!("single"));
                map.insert("total".to_string(), serde_json::json!(total));
                if let Some(r) = reason {
                    map.insert("reason".to_string(), serde_json::json!(r));
                }
                serde_json::Value::Object(map)
            }
            DiceRoll::Multi(values, reason) => {
                let total: i64 = values.iter().sum();
                let mut map = serde_json::Map::new();
                map.insert("type".to_string(), serde_json::json!("multi"));
                map.insert("values".to_string(), serde_json::json!(values));
                map.insert("total".to_string(), serde_json::json!(total));
                if let Some(r) = reason {
                    map.insert("reason".to_string(), serde_json::json!(r));
                }
                serde_json::Value::Object(map)
            }
        };

        serde_json::to_string_pretty(&value).map_err(crate::error::Error::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // DiceRoll::Display — Markdown output
    // -------------------------------------------------------------------------

    #[test]
    fn markdown_single_shows_total() {
        let output = DiceRoll::Single(42, None).to_string();
        assert!(output.contains("42"));
    }

    #[test]
    fn markdown_single_with_reason_shows_reason() {
        let output = DiceRoll::Single(17, Some("attack roll".to_string())).to_string();
        assert!(output.contains("17"));
        assert!(output.contains("attack roll"));
    }

    #[test]
    fn markdown_single_without_reason_has_no_parentheses() {
        assert!(!DiceRoll::Single(5, None).to_string().contains('('));
    }

    #[test]
    fn markdown_multi_shows_all_values() {
        let output = DiceRoll::Multi(vec![3, 5, 2], None).to_string();
        assert!(output.contains('3'));
        assert!(output.contains('5'));
        assert!(output.contains('2'));
    }

    #[test]
    fn markdown_multi_shows_sum_as_total() {
        // 3 + 5 + 2 = 10
        assert!(
            DiceRoll::Multi(vec![3, 5, 2], None)
                .to_string()
                .contains("10")
        );
    }

    #[test]
    fn markdown_multi_with_reason_shows_reason_and_total() {
        let output = DiceRoll::Multi(vec![4, 6], Some("damage".to_string())).to_string();
        assert!(output.contains("damage"));
        assert!(output.contains("10")); // 4 + 6
    }

    #[test]
    fn markdown_multi_without_reason_has_no_parentheses() {
        assert!(!DiceRoll::Multi(vec![1, 2], None).to_string().contains('('));
    }

    // -------------------------------------------------------------------------
    // DiceRoll::to_xml — LLM context
    // -------------------------------------------------------------------------

    #[test]
    fn xml_single_contains_type_and_total() {
        let xml = DiceRoll::Single(42, None).to_xml();
        assert!(xml.contains("type=\"single\""));
        assert!(xml.contains("<total>42</total>"));
    }

    #[test]
    fn xml_single_with_reason() {
        let xml = DiceRoll::Single(17, Some("attack roll".to_string())).to_xml();
        assert!(xml.contains("<reason>attack roll</reason>"));
    }

    #[test]
    fn xml_single_omits_reason_when_absent() {
        assert!(!DiceRoll::Single(5, None).to_xml().contains("<reason>"));
    }

    #[test]
    fn xml_multi_contains_type_values_and_total() {
        let xml = DiceRoll::Multi(vec![3, 5, 2], None).to_xml();
        assert!(xml.contains("type=\"multi\""));
        assert!(xml.contains("<values>3, 5, 2</values>"));
        assert!(xml.contains("<total>10</total>"));
    }

    #[test]
    fn xml_multi_with_reason() {
        let xml = DiceRoll::Multi(vec![4, 6], Some("damage".to_string())).to_xml();
        assert!(xml.contains("<reason>damage</reason>"));
    }

    #[test]
    fn xml_multi_omits_reason_when_absent() {
        assert!(
            !DiceRoll::Multi(vec![2, 3], None)
                .to_xml()
                .contains("<reason>")
        );
    }

    #[test]
    fn xml_is_well_formed() {
        let xml = DiceRoll::Single(7, None).to_xml();
        assert!(xml.starts_with("<roll"));
        assert!(xml.ends_with("</roll>"));
    }

    // -------------------------------------------------------------------------
    // DiceRoll::to_json — agent-skill output
    // -------------------------------------------------------------------------

    fn parse(roll: &DiceRoll) -> serde_json::Value {
        serde_json::from_str(&roll.to_json().expect("to_json failed")).expect("invalid JSON")
    }

    #[test]
    fn json_single_type_and_total() {
        let json = parse(&DiceRoll::Single(42, None));
        assert_eq!(json["type"], "single");
        assert_eq!(json["total"], 42);
    }

    #[test]
    fn json_single_with_reason() {
        let json = parse(&DiceRoll::Single(17, Some("attack roll".to_string())));
        assert_eq!(json["reason"], "attack roll");
    }

    #[test]
    fn json_single_omits_reason_when_absent() {
        let json = parse(&DiceRoll::Single(5, None));
        assert!(json.get("reason").is_none());
    }

    #[test]
    fn json_single_has_no_values_field() {
        let json = parse(&DiceRoll::Single(5, None));
        assert!(json.get("values").is_none());
    }

    #[test]
    fn json_multi_type_values_and_total() {
        let json = parse(&DiceRoll::Multi(vec![3, 5, 2], None));
        assert_eq!(json["type"], "multi");
        assert_eq!(json["values"], serde_json::json!([3, 5, 2]));
        assert_eq!(json["total"], 10);
    }

    #[test]
    fn json_multi_with_reason() {
        let json = parse(&DiceRoll::Multi(vec![4, 6], Some("damage".to_string())));
        assert_eq!(json["reason"], "damage");
        assert_eq!(json["total"], 10);
    }

    #[test]
    fn json_multi_omits_reason_when_absent() {
        let json = parse(&DiceRoll::Multi(vec![2, 3], None));
        assert!(json.get("reason").is_none());
    }

    #[test]
    fn json_multi_total_is_sum_of_values() {
        let json = parse(&DiceRoll::Multi(vec![1, 2, 3, 4], None));
        assert_eq!(json["total"], 10);
    }
}
