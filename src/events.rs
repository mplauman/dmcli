#[allow(dead_code)]
pub enum AppEvent {
    UserInput(String),
    AiResponse(String),
    AiThinking(String),
    AiError(String),
    CommandResult(String),
    CommandError(String),
    Exit,
}
