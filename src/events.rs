use crate::commands::DmCommand;

#[derive(Debug)]
pub enum AppEvent {
    UserCommand(DmCommand),
    UserAgent(String),
    AiResponse(String),
    AiThinking(String),
    AiError(String),
    AiComplete,
    #[allow(dead_code)]
    CommandResult(String),
    #[allow(dead_code)]
    CommandError(String),
    Exit,
}
