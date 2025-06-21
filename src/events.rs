use crate::commands::DmCommand;

#[derive(Debug)]
pub enum AppEvent {
    UserCommand(DmCommand),
    UserAgent(String),
    #[allow(dead_code)]
    AiResponse(String),
    #[allow(dead_code)]
    AiThinking(String),
    #[allow(dead_code)]
    AiError(String),
    #[allow(dead_code)]
    CommandResult(String),
    #[allow(dead_code)]
    CommandError(String),
    Exit,
}
