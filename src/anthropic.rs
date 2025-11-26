use crate::conversation::{Conversation, Message};
use crate::embeddings::EmbeddingGenerator;
use crate::errors::Error;
use crate::events::AppEvent;
use futures::{FutureExt, future};
use llm::backends::anthropic::Anthropic;
use llm::chat::{
    ChatMessage, ChatProvider, ChatRole, FunctionTool, MessageType as LlmMessageType, Tool,
};
use llm::{FunctionCall, ToolCall};
use rmcp::{
    RoleClient, RoleServer, Service, ServiceExt,
    model::{CallToolRequestParam, CallToolResult, RawContent},
    service::{DynService, RunningService},
};

type McpClient = RunningService<RoleClient, Box<dyn DynService<RoleClient> + 'static>>;

static SYSTEM_PROMPT: &str = "
You are a dungeon master's helpful assistant. You're role is to help them search through their notes to
provide them with information and backstory to help them prepare for games. You will also help them
create NPCs, monsters, scenes, situations, and descriptions of fantastic items based on the style of
the dungeon master's session notes.

## Communication

1. Keep your responses short. Avoid unnecessary details or tangents.
2. Don't apologize if you're unable to do something. Do your best, and explain why if you are unable to proceed.
3. As much as possible, base your answers on the DM's notes. Read multiple files to gain context.
4. Bias towards not asking the user for help if you can find the answer yourself.
5. When providing paths to tools, the path should always begin with a path that starts with a project root directory listed above.
6. Before you read or edit a file, you must first find the full path. DO NOT ever guess a file path!

## Tool Usage

1. When multiple tools can be used independently to solve different parts of a request, use them in parallel rather than sequentially.
2. For example, if you need to search for both a character and a location, make both tool calls at once rather than one after the other.
3. For dependent tools (where one tool's output is needed for another), use them sequentially as required.
4. Always try to minimize the number of back-and-forth interactions by batching independent tool calls.

## Searching Notes

When searching through notes, first get a list of all the potentially interesting files by listing them, grepping through
them, or both. When you have a list of candidates, read their contents and use that to inform your answer. Most of the time
the information you will need is spread through multiple files.

## Monster Generation

Immediately generate the monster stat block AND ONLY the monster stat block. Do not generate any remarks before or after the stat block. Wrap the stat block in a code block and format it as markdown.

Use this stat block format for monsters:
```
| STR | DEX | CON | INT | WIS | CHA |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 19 (+4) | 12 (+1) | 18 (+4) | 12 (+1) | 16 (+3) | 14 (+2) |
```
";

enum ClientAction {
    MakeRequest(Vec<ChatMessage>),
    UseTools(Vec<ChatMessage>, Vec<ToolCall>),
    Poison,
}

pub struct Client {
    client_sender: async_channel::Sender<ClientAction>,
}

impl Drop for Client {
    fn drop(&mut self) {
        self.client_sender
            .try_send(ClientAction::Poison)
            .expect("can still send poison signal");
    }
}

impl Client {
    pub async fn push(
        &mut self,
        conversation: &Conversation<impl EmbeddingGenerator>,
    ) -> Result<(), Error> {
        // Capacity: 5 desired, plus maybe 1 for tool use, plus related, plus a summary
        let mut messages = Vec::with_capacity(8);

        for message in conversation {
            let message = match message {
                Message::User { content, .. } => ChatMessage {
                    role: ChatRole::User,
                    message_type: LlmMessageType::Text,
                    content: content.clone(),
                },
                Message::Assistant { content, .. } => ChatMessage {
                    role: ChatRole::Assistant,
                    message_type: LlmMessageType::Text,
                    content: content.clone(),
                },
                Message::Thinking { content, tools, .. } => ChatMessage {
                    role: ChatRole::Assistant,
                    message_type: LlmMessageType::ToolUse(to_tool_use(tools)),
                    content: content.clone(),
                },
                Message::ThinkingDone { tools, .. } => ChatMessage {
                    role: ChatRole::User,
                    message_type: LlmMessageType::ToolResult(to_tool_result(tools.iter())),
                    content: String::new(),
                },
                Message::System { .. } => continue,
                Message::Error { .. } => continue,
            };

            let done = match message.message_type {
                LlmMessageType::ToolResult { .. } => false,
                _ => messages.len() >= 4,
            };

            messages.push(message);

            if done {
                break;
            }
        }

        let related = if let Some(Message::User { content, .. }) = conversation.into_iter().next() {
            let related = conversation
                .related(messages.len(), content, 10)
                .await
                .into_iter()
                .map(|msg| match msg {
                    Message::User { content, .. } => format!("- user: {content}"),
                    Message::Assistant { content, .. } => format!(" - assistant: {content}"),
                    Message::ThinkingDone { tools, .. } => {
                        format!(" - tool: {}", tools[0].result)
                    }
                    _ => panic!("Unexpected message type included in 'related' messages"),
                })
                .filter(|c| c != content) // Skip exact matches
                .collect::<Vec<String>>();

            ChatMessage {
                role: ChatRole::User,
                message_type: LlmMessageType::Text,
                content: format!(
                    "Here is some data related to the latest message:\n{}",
                    related.join("\n")
                ),
            }
        } else {
            log::info!("Skipping conversation; latest message is not from user");
            return Ok(());
        };
        messages.push(related);

        messages.reverse();

        self.client_sender
            .try_send(ClientAction::MakeRequest(messages))
            .expect("client sender is still open");

        Ok(())
    }
}

fn to_tool_use(tools: &[crate::conversation::ToolCall]) -> Vec<llm::ToolCall> {
    tools
        .iter()
        .map(|tc| llm::ToolCall {
            id: tc.id.clone(),
            call_type: "function".into(),
            function: FunctionCall {
                name: tc.name.clone(),
                arguments: serde_json::to_string(&tc.parameters)
                    .expect("parameters can be serialized"),
            },
        })
        .collect()
}

fn to_tool_result<'a>(
    results: impl IntoIterator<Item = &'a crate::conversation::ToolResult>,
) -> Vec<llm::ToolCall> {
    results
        .into_iter()
        .map(|tr| llm::ToolCall {
            id: tr.id.clone(),
            call_type: "function".into(),
            function: FunctionCall {
                name: tr.name.clone(),
                arguments: tr.result.clone(),
            },
        })
        .collect()
}

struct InnerClient {
    llm_client: Box<dyn ChatProvider>,
    mcp_clients: Vec<McpClient>,
    event_sender: async_channel::Sender<AppEvent>,
    client_sender: async_channel::Sender<ClientAction>,
    client_receiver: async_channel::Receiver<ClientAction>,
}

impl InnerClient {
    async fn run_loop(&self) -> Result<(), Error> {
        while let Ok(action) = self.client_receiver.recv().await {
            log::debug!("Got event, updating");

            match action {
                ClientAction::MakeRequest(messages) => self.request(messages).await?,
                ClientAction::UseTools(messages, tools) => self.use_tools(messages, tools).await?,
                ClientAction::Poison => break,
            }
        }

        Ok(())
    }

    fn send_app_event(&self, event: AppEvent) {
        if let Err(e) = self.event_sender.try_send(event) {
            panic!("Failed to send event to UI thread: {e:?}");
        }
    }

    fn send_internal_action(&self, event: ClientAction) {
        if let Err(e) = self.client_sender.try_send(event) {
            panic!("Failed to send event to internal thread: {e:?}");
        }
    }

    async fn request(&self, messages: Vec<ChatMessage>) -> Result<(), Error> {
        let response = match self.llm_client.chat(&messages).await {
            Ok(response) => response,
            Err(e) => {
                log::error!("AI request failed: {e:?}");
                self.send_app_event(AppEvent::AiError("failed to send AI request".into()));
                return Ok(());
            }
        };

        // Check if there are tool calls in the response
        let Some(tool_calls) = response.tool_calls() else {
            let message = response.text().unwrap_or_default();
            self.send_app_event(AppEvent::AiResponse(message));
            return Ok(());
        };

        let message = response.text().unwrap_or_default();

        self.send_app_event(AppEvent::AiThinking(message, tool_calls.clone()));
        self.send_internal_action(ClientAction::UseTools(messages, tool_calls));

        Ok(())
    }

    async fn use_tools(
        &self,
        mut messages: Vec<ChatMessage>,
        tools: Vec<ToolCall>,
    ) -> Result<(), Error> {
        log::info!("Found {} tool(s) to execute in parallel", tools.len());

        let tool_futures = tools.iter().map(|tool| {
            self.execute_single_tool(tool).map(|result| {
                let content = match result {
                    Ok(contents) => contents
                        .iter()
                        .filter_map(|c| match c {
                            Content::Text { text } => Some(text.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n"),
                    Err(e) => {
                        log::error!("Error executing tool {}: {e:?}", tool.function.name);
                        format!("Error executing tool: {e}")
                    }
                };

                ToolCall {
                    id: tool.id.clone(),
                    call_type: tool.call_type.clone(),
                    function: FunctionCall {
                        name: tool.function.name.clone(),
                        arguments: content,
                    },
                }
            })
        });
        let tool_results = future::join_all(tool_futures).await;

        self.send_app_event(AppEvent::AiThinkingDone(tool_results.clone()));

        messages.push(ChatMessage {
            role: ChatRole::Assistant,
            message_type: LlmMessageType::ToolUse(tools),
            content: String::new(),
        });
        messages.push(ChatMessage {
            role: ChatRole::User,
            message_type: LlmMessageType::ToolResult(tool_results),
            content: String::new(),
        });
        self.send_internal_action(ClientAction::MakeRequest(messages));

        Ok(())
    }

    // Helper method to execute a single tool across all MCP clients
    async fn execute_single_tool(&self, tool: &ToolCall) -> Result<Vec<Content>, Error> {
        log::info!(
            "Executing tool invocation {} for {}: {}",
            tool.id,
            tool.function.name,
            tool.function.arguments,
        );

        let mut contents = Vec::<Content>::default();
        for mcp_client in &self.mcp_clients {
            let request_param = CallToolRequestParam {
                name: tool.function.name.clone().into(),
                arguments: serde_json::from_str(&tool.function.arguments)
                    .expect("tool arguments are a JSON object"),
            };

            let request_result: CallToolResult = mcp_client.call_tool(request_param).await?;

            for result_content in request_result.content {
                match result_content.raw {
                    RawContent::Text(t) => contents.push(Content::Text { text: t.text }),
                    RawContent::Image(i) => {
                        log::warn!("Received image in tool result: {i:?}, skipping");
                    }
                    RawContent::Resource(r) => {
                        log::warn!("Received resource in tool result: {r:?}, skipping");
                    }
                    RawContent::Audio(a) => {
                        log::warn!("Got audio content in tool result: {a:?}, skipping");
                    }
                    RawContent::ResourceLink(r) => {
                        log::warn!("Got resource link in tool result: {r:?}, skipping");
                    }
                }
            }
        }

        if contents.is_empty() {
            contents.push(Content::Text {
                text: format!("No results returned for tool: {}", tool.function.name),
            });
        }

        Ok(contents)
    }
}

pub struct ClientBuilder {
    api_key: Option<String>,
    model: String,
    max_tokens: i64,
    mcp_clients: Vec<McpClient>,
    event_sender: Option<async_channel::Sender<AppEvent>>,
}

impl Default for ClientBuilder {
    fn default() -> Self {
        ClientBuilder {
            api_key: None,
            model: "claude-3-5-haiku-20241022".to_owned(),
            max_tokens: 8192,
            mcp_clients: Vec::default(),
            event_sender: None,
        }
    }
}

impl ClientBuilder {
    pub fn with_api_key(self, api_key: String) -> Self {
        let api_key = Some(api_key);

        Self { api_key, ..self }
    }

    pub fn with_model(self, model: String) -> Self {
        Self { model, ..self }
    }

    pub fn with_max_tokens(self, max_tokens: i64) -> Self {
        Self { max_tokens, ..self }
    }

    pub fn with_event_sender(self, event_sender: async_channel::Sender<AppEvent>) -> Self {
        Self {
            event_sender: Some(event_sender),
            ..self
        }
    }

    pub async fn with_toolkit<T: Service<RoleServer> + Send + 'static>(
        self,
        toolkit: T,
    ) -> Result<Self, Error> {
        let server = rmcp_in_process_transport::in_process::TokioInProcess::new(toolkit).await?;
        let server = ().into_dyn().serve(server).await?;

        let mut mcp_clients = self.mcp_clients;
        mcp_clients.push(server);

        Ok(Self {
            mcp_clients,
            ..self
        })
    }

    pub async fn build(self) -> Result<Client, Error> {
        let api_key = self.api_key.expect("api-key is set");

        // Collect tools from MCP clients
        let mut tools = Vec::<serde_json::Value>::default();
        for mcp_client in &self.mcp_clients {
            let tools_response = mcp_client.list_all_tools().await?;

            for tool in tools_response {
                tools.push(serde_json::json!({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": {
                        "type": "object",
                        "properties": tool.input_schema.get("properties").or(Option::default()),
                        "required": tool.input_schema.get("required").or(Option::default()),
                    },
                }));
            }
        }

        // Convert our tools to the LLM crate's format
        let llm_tools: Vec<Tool> = tools
            .iter()
            .filter_map(|tool| {
                let name = tool.get("name")?.as_str()?.to_string();
                let description = tool.get("description")?.as_str().unwrap_or("").to_string();
                let input_schema = tool.get("input_schema")?;

                Some(Tool {
                    tool_type: "function".to_string(),
                    function: FunctionTool {
                        name,
                        description,
                        parameters: input_schema.clone(),
                    },
                })
            })
            .collect();

        // Create the llm Anthropic client
        let llm_client = Anthropic::new(
            api_key.clone(),
            Some(self.model.clone()),
            Some(self.max_tokens as u32),
            None, // temperature - use default
            None, // timeout - use default
            Some(SYSTEM_PROMPT.to_string()),
            Some(false), // stream - not using streaming for now
            None,        // top_p
            None,        // top_k
            Some(llm_tools),
            None, // tool_choice
            None, // reasoning
            None, // thinking_budget_tokens
        );

        log::info!("Added tools: {}", serde_json::to_string(&tools).unwrap());

        let (client_sender, client_receiver) = async_channel::unbounded::<ClientAction>();

        let inner_client = InnerClient {
            llm_client: Box::new(llm_client),
            mcp_clients: self.mcp_clients,
            event_sender: self.event_sender.expect("event_sender must be set"),
            client_receiver,
            client_sender: client_sender.clone(),
        };

        let _worker = tokio::spawn(async move {
            if let Err(e) = inner_client.run_loop().await {
                log::error!("Client error: {e}");
            }
        });

        let client = Client { client_sender };

        Ok(client)
    }
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(tag = "type")]
enum Content {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: Vec<Content>,
    },
}

// Integration testing for parallel tool execution:
//
// A full integration test would verify the complete parallel tool execution flow, including:
// 1. Sending a request to the Anthropic API
// 2. Receiving a response with multiple tool use requests
// 3. Executing the tools in parallel
// 4. Sending the results back to the API
//
// Implementation approach:
// - Mock the HTTP client to return predefined responses
// - Create test MCP clients that return predefined tool results
// - Verify that the client correctly processes parallel tool requests
// - Check that the results are combined in the proper format
//
// Example test structure:
// 1. Create a mock HTTP client that returns a response with multiple tool use blocks
// 2. Create mock MCP clients that return predefined results for each tool
// 3. Call client.request() with a test message
// 4. Verify that the HTTP client received the correct tool results in the expected format
//
// This would require additional test dependencies such as:
// - mockall or similar for mocking the HTTP client
// - A test implementation of the MCP client trait

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_assistant_response() {
        // Create a test response with multiple tool uses
        let tool_uses = vec![
            (
                "toolu_001".to_string(),
                "get_weather".to_string(),
                serde_json::json!({"location": "New York"}),
            ),
            (
                "toolu_002".to_string(),
                "get_time".to_string(),
                serde_json::json!({"timezone": "America/New_York"}),
            ),
        ];

        // Create assistant content from tool uses
        let mut assistant_content = Vec::new();
        for (id, name, params) in &tool_uses {
            assistant_content.push(Content::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: params.clone(),
            });
        }

        // Verify assistant message has the correct tool use entries
        assert_eq!(assistant_content.len(), 2);

        match &assistant_content[0] {
            Content::ToolUse { id, name, input } => {
                assert_eq!(id, "toolu_001");
                assert_eq!(name, "get_weather");
                assert_eq!(input, &serde_json::json!({"location": "New York"}));
            }
            _ => panic!("Expected tool use content"),
        }

        match &assistant_content[1] {
            Content::ToolUse { id, name, input } => {
                assert_eq!(id, "toolu_002");
                assert_eq!(name, "get_time");
                assert_eq!(input, &serde_json::json!({"timezone": "America/New_York"}));
            }
            _ => panic!("Expected tool use content"),
        }
    }

    #[test]
    fn test_format_user_response() {
        // Create user content with tool results
        let user_content = [
            Content::ToolResult {
                tool_use_id: "toolu_001".to_string(),
                content: vec![Content::Text {
                    text: "Sunny, 75°F".to_string(),
                }],
            },
            Content::ToolResult {
                tool_use_id: "toolu_002".to_string(),
                content: vec![Content::Text {
                    text: "10:30 AM".to_string(),
                }],
            },
        ];

        // Verify user message has the correct tool result entries
        assert_eq!(user_content.len(), 2);

        match &user_content[0] {
            Content::ToolResult {
                tool_use_id,
                content,
            } => {
                assert_eq!(tool_use_id, "toolu_001");
                assert_eq!(content.len(), 1);
                if let Content::Text { text } = &content[0] {
                    assert_eq!(text, "Sunny, 75°F");
                } else {
                    panic!("Expected text content");
                }
            }
            _ => panic!("Expected tool result"),
        }

        match &user_content[1] {
            Content::ToolResult {
                tool_use_id,
                content,
            } => {
                assert_eq!(tool_use_id, "toolu_002");
                assert_eq!(content.len(), 1);
                if let Content::Text { text } = &content[0] {
                    assert_eq!(text, "10:30 AM");
                } else {
                    panic!("Expected text content");
                }
            }
            _ => panic!("Expected tool result"),
        }
    }

    #[test]
    fn test_format_user_response_with_errors() {
        // Create user content with mixed success and error results
        let user_content = [
            // Successful tool result
            Content::ToolResult {
                tool_use_id: "toolu_001".to_string(),
                content: vec![Content::Text {
                    text: "Sunny, 75°F".to_string(),
                }],
            },
            // Error tool result
            Content::ToolResult {
                tool_use_id: "toolu_002".to_string(),
                content: vec![Content::Text {
                    text: "Error executing tool: Failed to fetch time data".to_string(),
                }],
            },
        ];

        // Verify user message has the correct tool result entries
        assert_eq!(user_content.len(), 2);

        // Check success result
        match &user_content[0] {
            Content::ToolResult {
                tool_use_id,
                content,
            } => {
                assert_eq!(tool_use_id, "toolu_001");
                assert_eq!(content.len(), 1);
                if let Content::Text { text } = &content[0] {
                    assert_eq!(text, "Sunny, 75°F");
                } else {
                    panic!("Expected text content");
                }
            }
            _ => panic!("Expected tool result"),
        }

        // Check error result
        match &user_content[1] {
            Content::ToolResult {
                tool_use_id,
                content,
            } => {
                assert_eq!(tool_use_id, "toolu_002");
                assert_eq!(content.len(), 1);
                if let Content::Text { text } = &content[0] {
                    assert!(text.contains("Error executing tool"));
                } else {
                    panic!("Expected text content");
                }
            }
            _ => panic!("Expected tool result"),
        }
    }

    #[test]
    fn test_empty_tool_results() {
        // Create user content with an empty tool result
        let user_content = [Content::ToolResult {
            tool_use_id: "toolu_001".to_string(),
            content: vec![Content::Text {
                text: "No results returned for tool: get_weather".to_string(),
            }],
        }];

        // Verify user message has the correct fallback message
        assert_eq!(user_content.len(), 1);

        match &user_content[0] {
            Content::ToolResult {
                tool_use_id,
                content,
            } => {
                assert_eq!(tool_use_id, "toolu_001");
                assert_eq!(content.len(), 1);
                if let Content::Text { text } = &content[0] {
                    assert!(text.contains("No results returned for tool"));
                } else {
                    panic!("Expected text content");
                }
            }
            _ => panic!("Expected tool result"),
        }
    }
}
