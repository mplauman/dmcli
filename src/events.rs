use crate::commands::DmCommand;

#[derive(Debug)]
pub enum AppEvent {
    UserCommand(DmCommand),
    UserAgent(String),
    AiResponse(String),
    AiThinking(String, Vec<(String, String, serde_json::Value)>),
    AiCompact(usize, usize),
    AiError(String),
    #[allow(dead_code)]
    CommandResult(String),
    #[allow(dead_code)]
    CommandError(String),
    InputUpdated {
        line: String,
        cursor: usize,
    },
    WindowResized {
        width: u16,
        height: u16,
    },
    ScrollBack,
    ScrollForward,
    Exit,
}
