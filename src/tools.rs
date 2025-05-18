pub trait Toolkit {
    fn list_tools(&self) -> Vec<serde_json::Value>;
}
