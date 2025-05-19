#[derive(Hash, PartialEq, Eq)]
pub struct Tool {
    pub name: &'static str,
    pub description: &'static str,
    pub input_schema: serde_json::Value,
}

impl From<&Tool> for serde_json::Value {
    fn from(tool: &Tool) -> serde_json::Value {
        serde_json::json!({
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema
        })
    }
}

pub trait Toolkit {
    fn list_tools(&self) -> Vec<Tool>;
}
