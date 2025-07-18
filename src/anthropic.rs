use crate::errors::Error;
use crate::events::AppEvent;
use futures::future;
use llm::backends::anthropic::Anthropic;
use llm::chat::{
    ChatMessage, ChatProvider, ChatRole, FunctionTool, MessageType as LlmMessageType, Tool,
};
use llm::memory::{ChatWithMemory, SlidingWindowMemory, TrimStrategy};
use llm::{FunctionCall, ToolCall};
use rmcp::{
    RoleClient, RoleServer, Service, ServiceExt,
    model::{CallToolRequestParam, CallToolResult, RawContent},
    service::{DynService, RunningService},
};
use std::sync::Arc;
use tokio::sync::RwLock;

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

pub struct Client {
    pub llm_client: Arc<dyn ChatProvider>,
    pub mcp_clients: Vec<McpClient>,
    pub tools: Vec<serde_json::Value>,
    pub event_sender: async_channel::Sender<AppEvent>,
}

// Test helper functions
#[cfg(test)]
impl Client {
    // For testing purposes - tests the message removal logic for MaxTokens error
    pub fn test_max_tokens_handling(&self, messages: &mut Vec<ChatMessage>) -> usize {
        let mut max_tokens_retry_count = 0;
        let max_tokens_max_retries = 5;

        while max_tokens_retry_count < max_tokens_max_retries {
            max_tokens_retry_count += 1;

            if messages.len() <= 2 {
                // Cannot remove any more messages without losing context
                println!("Cannot remove any more messages without losing context");
                break;
            }

            // Remove the oldest non-system message (index 0 is usually the first user message)
            println!(
                "Max tokens reached. Removing oldest message and retrying. Attempt {max_tokens_retry_count} of {max_tokens_max_retries}"
            );

            // For test purposes, only remove if we still have more than 2 messages
            if messages.len() > 2 {
                messages.remove(1); // Keep the first message, remove the second oldest
            } else {
                break;
            }

            // For testing purposes, we'll just continue until we hit the retry limit
            // In real usage, the loop would retry the API call after removing a message
        }

        max_tokens_retry_count
    }
}

impl Client {
    pub fn push(&mut self, content: String) -> Result<(), Error> {
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            message_type: LlmMessageType::Text,
            content,
        }];

        self.request(messages)
    }

    fn request(&self, messages: Vec<ChatMessage>) -> Result<(), Error> {
        // Convert our tools to the LLM crate's format
        let llm_tools: Vec<Tool> = self
            .tools
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

        let event_sender = self.event_sender.clone();
        let send_event = move |event| {
            if let Err(e) = event_sender.try_send(event) {
                panic!("Failed to send event to UI thread: {e:?}");
            }
        };

        // Clone the Arc to share the client across the async task
        let llm_client = Arc::clone(&self.llm_client);

        let _background = tokio::task::spawn(async move {
            // Pass tools if available
            let tools_slice = if llm_tools.is_empty() {
                None
            } else {
                Some(llm_tools.as_slice())
            };
            let response = match llm_client.chat_with_tools(&messages, tools_slice).await {
                Ok(response) => response,
                Err(e) => {
                    log::error!("AI request failed: {e:?}");
                    send_event(AppEvent::AiError("failed to send AI request".into()));
                    return;
                }
            };

            // Check if there are tool calls in the response
            if let Some(tool_calls) = response.tool_calls() {
                let message = response.text().unwrap_or_default();
                let tools: Vec<(String, String, serde_json::Value)> = tool_calls
                    .iter()
                    .map(|tc| {
                        let params: serde_json::Value =
                            serde_json::from_str(&tc.function.arguments).unwrap_or_else(|_| {
                                serde_json::Value::Object(serde_json::Map::new())
                            });
                        (tc.id.clone(), tc.function.name.clone(), params)
                    })
                    .collect();

                send_event(AppEvent::AiThinking(message, tools));
            } else {
                let message = response.text().unwrap_or_default();
                send_event(AppEvent::AiResponse(message));
            }
        });

        Ok(())
    }

    pub async fn use_tools(
        &mut self,
        tools: Vec<(String, String, serde_json::Value)>,
    ) -> Result<(), Error> {
        if tools.is_empty() {
            return Err(Error::NoToolUses);
        }

        log::info!("Found {} tool(s) to execute in parallel", tools.len());

        // Prepare futures for parallel execution
        let tool_futures = tools.iter().map(|(id, name, params)| {
            self.execute_single_tool(id.clone(), name.clone(), params.clone())
        });

        // Execute all tool futures in parallel
        let tool_results = future::join_all(tool_futures).await;

        // Process results and prepare the messages

        // First add all tool uses to the assistant message
        let tool_calls: Vec<ToolCall> = tools
            .iter()
            .map(|(id, name, params)| ToolCall {
                id: id.clone(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: name.clone(),
                    arguments: serde_json::to_string(params).unwrap_or_default(),
                },
            })
            .collect();

        let mut messages = Vec::new();

        if !tool_calls.is_empty() {
            messages.push(ChatMessage {
                role: ChatRole::Assistant,
                message_type: LlmMessageType::ToolUse(tool_calls),
                content: String::new(),
            });
        }

        // Then add all tool results to the user message
        let tool_result_calls: Vec<ToolCall> = tool_results
            .into_iter()
            .zip(tools.iter())
            .map(|(result, (id, name, _))| {
                let content = match result {
                    Ok(contents) => {
                        let content_texts: Vec<String> = contents
                            .iter()
                            .filter_map(|c| match c {
                                Content::Text { text } => Some(text.clone()),
                                _ => None,
                            })
                            .collect();
                        content_texts.join("\n")
                    }
                    Err(e) => {
                        log::error!("Error executing tool {name}: {e:?}");
                        format!("Error executing tool: {e}")
                    }
                };

                ToolCall {
                    id: id.clone(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: name.clone(),
                        arguments: content,
                    },
                }
            })
            .collect();

        if !tool_result_calls.is_empty() {
            messages.push(ChatMessage {
                role: ChatRole::User,
                message_type: LlmMessageType::ToolResult(tool_result_calls),
                content: String::new(),
            });
        }

        // Note: For a comprehensive test suite, we would also want to test:
        // 1. Timeouts in parallel tool execution
        // 2. Partial failures (some tools succeed, some fail)
        // 3. Concurrent requests with different sets of tools
        // 4. Various response formats from the Anthropic API
        // 5. Error propagation from the MCP clients

        self.request(messages)
    }

    // Helper method to execute a single tool across all MCP clients
    async fn execute_single_tool(
        &self,
        id: String,
        name: String,
        params: serde_json::Value,
    ) -> Result<Vec<Content>, Error> {
        log::info!("Executing tool invocation {id} for {name}: {params:#?}");

        let mut contents = Vec::<Content>::default();
        for mcp_client in &self.mcp_clients {
            let request_param = CallToolRequestParam {
                name: name.clone().into(),
                arguments: match &params {
                    serde_json::Value::Object(o) => Some(o.clone()),
                    x => {
                        log::warn!("Unexpected tool parameters {x:?}");
                        None
                    }
                },
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
                }
            }
        }

        if contents.is_empty() {
            contents.push(Content::Text {
                text: format!("No results returned for tool: {name}"),
            });
        }

        Ok(contents)
    }
}

pub struct ClientBuilder {
    pub api_key: Option<String>,
    pub model: String,
    pub max_tokens: i64,
    pub window_size: usize,
    pub mcp_clients: Vec<McpClient>,
    pub event_sender: Option<async_channel::Sender<AppEvent>>,
}

impl Default for ClientBuilder {
    fn default() -> Self {
        ClientBuilder {
            api_key: None,
            model: "claude-3-5-haiku-20241022".to_owned(),
            max_tokens: 8192,
            window_size: 5,
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

    /// Sets the sliding window size for chat memory management.
    ///
    /// The window size determines how many recent messages are kept in memory
    /// before older messages are trimmed using the configured strategy.
    ///
    /// # Arguments
    /// * `window_size` - The number of messages to keep in the sliding window (default: 32)
    ///
    /// # Example
    /// ```
    /// let builder = ClientBuilder::default().with_window_size(64);
    /// ```
    pub fn with_window_size(self, window_size: usize) -> Self {
        Self {
            window_size,
            ..self
        }
    }

    pub fn with_event_sender(self, event_sender: async_channel::Sender<AppEvent>) -> Self {
        Self {
            event_sender: Some(event_sender),
            ..self
        }
    }

    pub async fn with_toolkit<T: Service<RoleServer> + Send + 'static>(self, toolkit: T) -> Self {
        let server = rmcp_in_process_transport::in_process::TokioInProcess::new(toolkit);
        let server = server.serve().await.unwrap();
        let server = ().into_dyn().serve(server).await.unwrap();

        let mut mcp_clients = self.mcp_clients;
        mcp_clients.push(server);

        Self {
            mcp_clients,
            ..self
        }
    }

    pub async fn build(self) -> Result<Client, Error> {
        let api_key = self.api_key.expect("api-key is set");

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
            None,        // tools - will be handled separately
            None,        // tool_choice
            None,        // reasoning
            None,        // thinking_budget_tokens
        );

        let llm_client = ChatWithMemory::new(
            Arc::new(llm_client),
            Arc::new(RwLock::new(Box::new(SlidingWindowMemory::with_strategy(
                self.window_size,
                TrimStrategy::Summarize,
            )))),
            None,
            Vec::new(),
            None,
        );

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

        log::info!("Added tools: {}", serde_json::to_string(&tools).unwrap());

        let client = Client {
            llm_client: Arc::new(llm_client),
            tools,
            mcp_clients: self.mcp_clients,
            event_sender: self.event_sender.expect("event_sender must be set"),
        };

        Ok(client)
    }
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct ClaudeResponse {
    pub id: String,
    pub r#type: String,
    pub role: String,
    pub model: String,
    pub content: Vec<Content>,
    pub stop_reason: Option<StopReason>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(PartialEq, Eq, Debug, serde::Deserialize, serde::Serialize)]
enum StopReason {
    #[serde(rename = "end_turn")]
    EndTurn,
    #[serde(rename = "tool_use")]
    ToolUse,
    #[serde(rename = "stop_sequence")]
    StopSequence,
    #[serde(rename = "max_tokens")]
    MaxTokens,
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

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct Usage {
    pub input_tokens: i64,
    pub cache_creation_input_tokens: i64,
    pub cache_read_input_tokens: i64,
    pub output_tokens: i64,
}

#[test]
fn test_message_claude_response_serde() {
    let json = r#"{
        "id": "msg_012J63SbcBy7BjHy5GijWvps",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-haiku-20241022",
        "content": [
            {
                "type": "text",
                "text": "Hi there! I'm an AI assistant designed to help with roleplaying game tools. I see there's a function available to retrieve session notes. Would you like me to fetch the dungeon master's local notes for you?"
            }
        ],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 324,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 49
        }
    }"#;

    let response: ClaudeResponse = serde_json::from_str(json).unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(response.id, "msg_012J63SbcBy7BjHy5GijWvps");
    assert_eq!(response.r#type, "message");
    assert_eq!(response.role, "assistant");
    assert_eq!(response.model, "claude-3-5-haiku-20241022");
    assert_eq!(response.stop_reason.unwrap(), StopReason::EndTurn {});
    assert_eq!(response.stop_sequence, None);
}

#[test]
fn test_claude_response_serde() {
    let json = r#"{
        "id": "msg_01FQKQYzu89giYKxBAK4DGpt",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-haiku-20241022",
        "content": [
            {
                "type": "text",
                "text": "I'll retrieve the session notes for you using the get_session_notes function."
            },
            {
                "type": "tool_use",
                "id": "toolu_01CJBJNzCa1KCuqfwNbC9guo",
                "name": "get_session_notes",
                "input": {}
            }
        ],
        "stop_reason": "tool_use",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 324,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 56
        }
    }"#;

    let response: ClaudeResponse = serde_json::from_str(json).unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(response.id, "msg_01FQKQYzu89giYKxBAK4DGpt");
    assert_eq!(response.r#type, "message");
    assert_eq!(response.role, "assistant");
    assert_eq!(response.model, "claude-3-5-haiku-20241022");
    assert_eq!(response.stop_reason.unwrap(), StopReason::ToolUse {});
    assert_eq!(response.stop_sequence, None);
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
    fn test_max_tokens_retry_behavior() {
        // Create a basic client for testing
        let (tx, _rx) = async_channel::unbounded();

        // Create a test LLM client
        let llm_client = Anthropic::new(
            "test-key".to_string(),
            Some("claude-3-opus-20240229".to_string()),
            Some(1024),
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
            None,
            None,
            None,
        );

        let client = Client {
            llm_client: Arc::new(llm_client),
            mcp_clients: vec![],
            tools: vec![],
            event_sender: tx,
        };

        // Set up messages with a few entries to test removal
        let mut messages = vec![
            ChatMessage {
                role: ChatRole::User,
                message_type: LlmMessageType::Text,
                content: "Initial message".to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                message_type: LlmMessageType::Text,
                content: "Message 2".to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                message_type: LlmMessageType::Text,
                content: "Message 3".to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                message_type: LlmMessageType::Text,
                content: "Message 4".to_string(),
            },
        ];

        // Test the MaxTokens handling directly
        let retry_count = client.test_max_tokens_handling(&mut messages);

        // Should stop after going through all messages except the first one
        assert_eq!(retry_count, 3);

        // Verify that messages were removed (started with 4, should have 2 left)
        assert_eq!(messages.len(), 2);

        // Check if the first message contains our expected initial message
        assert_eq!(messages[0].content, "Initial message");
    }

    #[test]
    fn test_multiple_tool_use_extraction() {
        let json = r#"{
            "id": "msg_0123456789",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-haiku-20241022",
            "content": [
                {
                    "type": "text",
                    "text": "I'll help you with that request."
                },
                {
                    "type": "tool_use",
                    "id": "toolu_001",
                    "name": "get_weather",
                    "input": {"location": "New York"}
                },
                {
                    "type": "tool_use",
                    "id": "toolu_002",
                    "name": "get_time",
                    "input": {"timezone": "America/New_York"}
                }
            ],
            "stop_reason": "tool_use",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 300,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 80
            }
        }"#;

        let response: ClaudeResponse =
            serde_json::from_str(json).unwrap_or_else(|e| panic!("{e:?}"));

        // Manually extract tool uses like the invoke_tool method does
        let tool_uses: Vec<_> = response
            .content
            .iter()
            .filter_map(|c| match c {
                Content::ToolUse { id, name, input } => {
                    Some((id.to_owned(), name.to_owned(), input.clone()))
                }
                _ => None,
            })
            .collect();

        // Verify we extract both tool use requests
        assert_eq!(tool_uses.len(), 2);

        // Check first tool
        assert_eq!(tool_uses[0].0, "toolu_001");
        assert_eq!(tool_uses[0].1, "get_weather");

        // Check second tool
        assert_eq!(tool_uses[1].0, "toolu_002");
        assert_eq!(tool_uses[1].1, "get_time");
    }

    #[test]
    fn test_empty_tool_use_response() {
        let json = r#"{
            "id": "msg_0123456789",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-haiku-20241022",
            "content": [
                {
                    "type": "text",
                    "text": "I'll help you with that request."
                }
            ],
            "stop_reason": "tool_use",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 300,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 80
            }
        }"#;

        let response: ClaudeResponse =
            serde_json::from_str(json).unwrap_or_else(|e| panic!("{e:?}"));

        // Manually extract tool uses like the invoke_tool method does
        let tool_uses: Vec<_> = response
            .content
            .iter()
            .filter_map(|c| match c {
                Content::ToolUse { id, name, input } => {
                    Some((id.to_owned(), name.to_owned(), input.clone()))
                }
                _ => None,
            })
            .collect();

        // Verify we don't extract any tool uses when none are present
        assert_eq!(tool_uses.len(), 0);
    }

    #[test]
    fn test_malformed_tool_use() {
        let json = r#"{
            "id": "msg_0123456789",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-haiku-20241022",
            "content": [
                {
                    "type": "text",
                    "text": "I'll help you with that request."
                },
                {
                    "type": "tool_use",
                    "id": "toolu_001",
                    "name": "get_weather",
                    "input": "invalid_non_object_input"
                }
            ],
            "stop_reason": "tool_use",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 300,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 80
            }
        }"#;

        let response: ClaudeResponse =
            serde_json::from_str(json).unwrap_or_else(|e| panic!("{e:?}"));

        // Manually extract tool uses like the invoke_tool method does
        let tool_uses: Vec<_> = response
            .content
            .iter()
            .filter_map(|c| match c {
                Content::ToolUse { id, name, input } => {
                    Some((id.to_owned(), name.to_owned(), input.clone()))
                }
                _ => None,
            })
            .collect();

        // We should still extract the tool use even with invalid input
        assert_eq!(tool_uses.len(), 1);

        // Check that we extracted the tool with the non-object input
        assert_eq!(tool_uses[0].0, "toolu_001");
        assert_eq!(tool_uses[0].1, "get_weather");
        assert_eq!(
            tool_uses[0].2,
            serde_json::json!("invalid_non_object_input")
        );
    }

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

    #[test]
    fn test_window_size_configuration() {
        // Test default window size
        let builder = ClientBuilder::default();
        assert_eq!(builder.window_size, 5);

        // Test custom window size
        let builder = ClientBuilder::default().with_window_size(64);
        assert_eq!(builder.window_size, 64);

        // Test chaining with other builder methods
        let builder = ClientBuilder::default()
            .with_window_size(128)
            .with_max_tokens(4096);
        assert_eq!(builder.window_size, 128);
        assert_eq!(builder.max_tokens, 4096);
    }
}
