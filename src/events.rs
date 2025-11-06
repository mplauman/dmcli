use crate::commands::DmCommand;

#[derive(Debug)]
pub enum AppEvent {
    UserCommand(DmCommand),
    UserAgent(String),
    AiResponse(String),
    AiThinking(String, Vec<llm::ToolCall>),
    AiThinkingDone(Vec<llm::ToolCall>),
    AiError(String),
    InputUpdated { line: String, cursor: usize },
    WindowResized { width: u16, height: u16 },
    ScrollBack,
    ScrollForward,
    Exit,
}
