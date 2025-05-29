use std::collections::HashMap;
use std::rc::Rc;

use crate::errors::Error;
use crate::tools::{Tool, Toolkit};

static SYSTEM_PROMPT: &str = "
You are a dungeon master's helpful assistant. You're role is to help them search through their notes to
provide them with information and backstory to help them prepare for games. You will also help them
create NPCs, monsters, scenes, situations, and descriptions of fantastic items based on the style of
the dungeon master's session notes.

## Communication

1. Keep your responses short. Avoid unnecessary details or tangents.
2. Don't apologize if you're unable to do something. Do your best, and explain why if you are unable to proceed.
3. If appropriate, use tool calls to explore the DM's notes.
4. Bias towards not asking the user for help if you can find the answer yourself.
5. When providing paths to tools, the path should always begin with a path that starts with a project root directory listed above.
6. Before you read or edit a file, you must first find the full path. DO NOT ever guess a file path!

You are being tasked with providing a response, but you have no ability to use tools or to read or write any aspect of the user's system (other than any context the user might have provided to you).

As such, if you need the user to perform any actions for you, you must request them explicitly. Bias towards giving a response to the best of your ability, and then making requests for the user to take action (e.g. to give you more context) only optionally.

The one exception to this is if the user references something you don't know about - for example, the name of a source code file, function, type, or other piece of code that you have no awareness of. In this case, you MUST NOT MAKE SOMETHING UP, or assume you know what that thing is or how it works. Instead, you must ask the user for clarification rather than giving a response.

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
    pub model: String,
    pub endpoint: String,
    pub client: reqwest::Client,
    pub tools: HashMap<Tool, Rc<Box<dyn Toolkit>>>,
}

impl Client {
    pub async fn request(&self, messages: &mut Vec<Message>) -> Result<(), Error> {
        loop {
            let body = serde_json::json!({
                "model": self.model.as_str(),
                "max_tokens": 1024,
                "system": SYSTEM_PROMPT,
                "tools": self.tools.keys().map(|t| t.into()).collect::<Vec<serde_json::Value>>(),
                "messages": messages,
            });

            log::trace!("AGENT REQUEST >>> {}", serde_json::to_string(&body)?);

            let request = self
                .client
                .post(self.endpoint.as_str())
                .json(&body)
                .build()?;

            let response: ClaudeResponse = self
                .client
                .execute(request)
                .await
                .unwrap()
                .json()
                .await
                .unwrap();

            log::trace!("AGENT RESPONSE <<< {}", serde_json::to_string(&response)?);

            match response.stop_reason {
                Some(StopReason::EndTurn) => {
                    println!(
                        "{}",
                        response
                            .extract_message()
                            .unwrap_or("<expected message not found>".to_owned())
                    );
                    break;
                }
                Some(StopReason::ToolUse) => {
                    println!(
                        ":thonking: {}",
                        response
                            .extract_message()
                            .unwrap_or("<no tool message>".to_owned())
                    );

                    let (assistant, user) = self.invoke_tool(response).await?;
                    messages.push(assistant);
                    messages.push(user);
                }
                Some(StopReason::StopSequence) => {
                    println!(":panic: How do I handle this: {:?}", response);
                    break;
                }
                Some(StopReason::MaxTokens) => {
                    println!(":panic: How do I handle this: {:?}", response);
                    break;
                }
                None => {
                    panic!("No stop reason was provided in {:?}", response);
                }
            };
        }

        Ok(())
    }

    async fn invoke_tool(&self, response: ClaudeResponse) -> Result<(Message, Message), Error> {
        let (id, name, params) = response
            .content
            .iter()
            .find_map(|c| match c {
                Content::ToolUse { id, name, input } => Some((id, name, input)),
                _ => None,
            })
            .expect("a tool is provided for execution");

        let toolkit = self
            .tools
            .iter()
            .find_map(|t| if t.0.name == name { Some(t.1) } else { None })
            .expect("a toolkit should exist");

        let response = toolkit.run_tool(name, params);

        let assistant = Message {
            role: Role::Assistant,
            content: vec![Content::ToolUse {
                id: id.to_owned(),
                name: name.to_owned(),
                input: params.clone(),
            }],
        };

        let user = Message {
            role: Role::User,
            content: vec![Content::ToolResult {
                tool_use_id: id.to_owned(),
                content: vec![Content::Text {
                    text: serde_json::to_string(&response).expect("response can be serialize"),
                }],
            }],
        };

        Ok((assistant, user))
    }
}

pub struct ClientBuilder {
    pub api_key: Option<String>,
    pub model: String,
    pub version: String,
    pub endpoint: String,
    pub toolkits: Vec<Box<dyn Toolkit>>,
}

impl Default for ClientBuilder {
    fn default() -> Self {
        ClientBuilder {
            api_key: None,
            model: "claude-3-5-haiku-20241022".to_owned(),
            version: "2023-06-01".to_owned(),
            endpoint: "https://api.anthropic.com/v1/messages".to_owned(),
            toolkits: Default::default(),
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

    pub fn with_toolkit<T: crate::tools::Toolkit + 'static>(self, toolkit: T) -> Self {
        let mut toolkits = self.toolkits;
        toolkits.push(Box::new(toolkit));

        Self { toolkits, ..self }
    }

    pub async fn build(self) -> Result<Client, Error> {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "x-api-key",
            self.api_key
                .expect("api-key is set")
                .parse()
                .expect("api-key is an HTTP header"),
        );
        headers.insert(
            "anthropic-version",
            self.version.parse().expect("version is an HTTP header"),
        );
        headers.insert(
            "content-type",
            "application/json"
                .parse()
                .expect("content-type is an HTTP header"),
        );

        let client_tools = self
            .toolkits
            .into_iter()
            .flat_map(|tk| {
                let tk = Rc::new(tk);

                tk.list_tools()
                    .into_iter()
                    .map(move |t| (t, Rc::clone(&tk)))
            })
            .collect::<HashMap<_, _>>();

        let client = Client {
            model: self.model,
            endpoint: self.endpoint,
            tools: client_tools,
            client: reqwest::ClientBuilder::default()
                .default_headers(headers)
                .build()?,
        };

        Ok(client)
    }
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct Message {
    role: Role,
    content: Vec<Content>,
}

impl Message {
    pub fn user(text: String) -> Message {
        Message {
            role: Role::User,
            content: vec![Content::Text { text }],
        }
    }
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
enum Role {
    #[serde(rename = "assistant")]
    Assistant,

    #[serde(rename = "user")]
    User,
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

impl ClaudeResponse {
    fn extract_message(&self) -> Option<String> {
        self.content.iter().find_map(|c| match c {
            Content::Text { text } => Some(text.to_owned()),
            _ => None,
        })
    }
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

    let response: ClaudeResponse = serde_json::from_str(json).unwrap_or_else(|e| panic!("{:?}", e));
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

    let response: ClaudeResponse = serde_json::from_str(json).unwrap_or_else(|e| panic!("{:?}", e));
    assert_eq!(response.id, "msg_01FQKQYzu89giYKxBAK4DGpt");
    assert_eq!(response.r#type, "message");
    assert_eq!(response.role, "assistant");
    assert_eq!(response.model, "claude-3-5-haiku-20241022");
    assert_eq!(response.stop_reason.unwrap(), StopReason::ToolUse {});
    assert_eq!(response.stop_sequence, None);
}
