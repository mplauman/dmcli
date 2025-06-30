use crate::errors::Error;
use crate::events::AppEvent;

use futures::future;
use llm::{
    builder::{FunctionBuilder, LLMBackend, LLMBuilder, ParamBuilder},
    chat::{ChatMessage, ChatRole, MessageType},
    LLMProvider,
};
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

pub struct Client {
    pub llm: Box<dyn LLMProvider>,
    pub mcp_clients: Vec<McpClient>,
    pub chat_history: Vec<ChatMessage>,
    pub event_sender: async_channel::Sender<AppEvent>,
    pub model: String,
    pub max_tokens: i64,
}

impl Client {
    fn send_event(&self, event: AppEvent) {
        if let Err(e) = self.event_sender.try_send(event) {
            panic!("Failed to send event to UI thread: {:?}", e);
        }
    }

    pub fn clear(&mut self) {
        self.chat_history.clear();
    }

    pub fn compact(&mut self, attempt: usize, max_attempts: usize) -> Result<(), Error> {
        if attempt >= max_attempts {
            self.send_event(AppEvent::AiError(format!(
                "Max retry attempts reached ({}) after removing oldest messages",
                attempt
            )));
            return Ok(());
        }

        if self.chat_history.len() <= 2 {
            self.send_event(AppEvent::AiError(
                "Cannot remove any more messages without losing context".to_string(),
            ));
            return Ok(());
        }

        self.chat_history.remove(1);
        self.request(attempt + 1)
    }

    pub fn push(&mut self, content: String) -> Result<(), Error> {
        self.chat_history.push(ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content,
        });
        self.request(0)
    }

    fn request(&mut self, _attempt: usize) -> Result<(), Error> {
        log::debug!("LLM REQUEST >>> {:#?}", self.chat_history);

        // For now, just send a simple response to test the integration
        let event_sender = self.event_sender.clone();
        let send_event = move |event| {
            if let Err(e) = event_sender.try_send(event) {
                panic!("Failed to send event to UI thread: {:?}", e);
            }
        };

        // Placeholder implementation - will be improved later
        send_event(AppEvent::AiResponse("LLM integration working! (Basic implementation)".to_string()));

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
        let mut tool_messages = Vec::new();

        // Add tool results to the conversation
        for (result, (_id, name, _)) in tool_results.into_iter().zip(tools.iter()) {
            let result_text = match result {
                Ok(contents) => {
                    contents.iter()
                        .map(|content| match content {
                            Content::Text { text } => text.clone(),
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                }
                Err(e) => {
                    log::error!("Error executing tool {name}: {e:?}");
                    format!("Error executing tool: {e}")
                }
            };

            // Add the tool result as a system message
            tool_messages.push(ChatMessage {
                role: ChatRole::Assistant,
                message_type: MessageType::Text,
                content: format!("Tool '{}' result: {}", name, result_text),
            });
        }

        // Add tool results to chat history
        self.chat_history.extend(tool_messages);

        // Continue the conversation
        self.request(0)
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

// Compatibility types to maintain the same interface
#[derive(Debug)]
enum Content {
    Text { text: String },
}

pub struct ClientBuilder {
    pub api_key: Option<String>,
    pub model: String,
    pub backend: LLMBackend,
    pub mcp_clients: Vec<McpClient>,
    pub max_tokens: i64,
    pub event_sender: Option<async_channel::Sender<AppEvent>>,
}

impl Default for ClientBuilder {
    fn default() -> Self {
        ClientBuilder {
            api_key: None,
            model: "claude-3-5-haiku-20241022".to_owned(),
            backend: LLMBackend::Anthropic,
            mcp_clients: Vec::default(),
            max_tokens: 8192,
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
        let backend = self.backend.clone(); // Clone before use
        
        // Create LLM builder
        let mut llm_builder = LLMBuilder::new()
            .backend(self.backend) // Use original value here
            .api_key(api_key)
            .model(&self.model)
            .max_tokens(self.max_tokens as u32);

        // Add system prompt if supported by the backend
        if matches!(backend, LLMBackend::Anthropic | LLMBackend::OpenAI) {
            llm_builder = llm_builder.system(SYSTEM_PROMPT);
        }

        // Convert MCP tools to LLM functions
        for mcp_client in &self.mcp_clients {
            let tools_response = mcp_client.list_all_tools().await?;

            for tool in tools_response {
                let mut function_builder = FunctionBuilder::new(tool.name.as_ref())
                    .description(tool.description.as_ref());

                // Convert MCP tool schema to LLM function parameters
                if let Some(properties) = tool.input_schema.get("properties") {
                    if let Some(props_obj) = properties.as_object() {
                        for (param_name, param_def) in props_obj {
                            if let Some(param_obj) = param_def.as_object() {
                                let param_type = param_obj.get("type")
                                    .and_then(|t| t.as_str())
                                    .unwrap_or("string");
                                let param_desc = param_obj.get("description")
                                    .and_then(|d| d.as_str())
                                    .unwrap_or("");

                                function_builder = function_builder.param(
                                    ParamBuilder::new(param_name)
                                        .type_of(param_type)
                                        .description(param_desc)
                                );
                            }
                        }
                    }
                }

                // Add required parameters
                if let Some(required) = tool.input_schema.get("required") {
                    if let Some(required_array) = required.as_array() {
                        let required_params: Vec<String> = required_array
                            .iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect();
                        function_builder = function_builder.required(required_params);
                    }
                }

                llm_builder = llm_builder.function(function_builder);
            }
        }

        let llm = llm_builder.build()
            .map_err(|e| Error::Generic(format!("Failed to build LLM client: {}", e)))?;

        log::info!("Built LLM client with {} MCP clients", self.mcp_clients.len());

        let client = Client {
            llm,
            mcp_clients: self.mcp_clients,
            chat_history: Vec::new(),
            event_sender: self.event_sender.expect("event_sender must be set"),
            model: self.model,
            max_tokens: self.max_tokens,
        };

        Ok(client)
    }
}