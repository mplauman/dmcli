use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use crate::errors::Error;
use crate::tools::{Tool, Toolkit};

pub struct Client {
    pub model: String,
    pub endpoint: String,
    pub client: reqwest::Client,
    pub tools: HashMap<Tool, Rc<Box<dyn Toolkit>>>,
}

impl Client {
    pub async fn request<T: serde::Serialize + ?Sized>(
        &self,
        messages: &T,
    ) -> Result<String, Error> {
        let mut message_queue = VecDeque::<serde_json::Value>::default();

        message_queue.push_back(serde_json::json!({
            "model": self.model.as_str(),
            "max_tokens": 1024,
            "tools": self.tools.keys().map(|t| t.into()).collect::<Vec<serde_json::Value>>(),
            "messages": messages,
        }));

        while let Some(message) = message_queue.pop_front() {
            println!("Request: {:?}", message);

            let request = self
                .client
                .post(self.endpoint.as_str())
                .json(&message)
                .build()?;

            let response: ClaudeResponse = self
                .client
                .execute(request)
                .await
                .unwrap()
                .json()
                .await
                .unwrap();

            println!("Response: {:?}", response);

            match response.stop_reason {
                Some(StopReason::EndTurn) => {
                    return Ok(response
                        .extract_message()
                        .unwrap_or("<expected message not found>".to_owned()));
                }
                Some(StopReason::ToolUse) => {
                    println!(
                        ":thonking: {}",
                        response
                            .extract_message()
                            .unwrap_or("<no tool message>".to_owned())
                    );

                    message_queue.push_back(self.invoke_tool(response).await?);
                }
                Some(StopReason::StopSequence) => {
                    println!(":panic: How do I handle this: {:?}", response);
                    return Ok("<stop sequence hit>".to_owned());
                }
                Some(StopReason::MaxTokens) => {
                    println!(":panic: How do I handle this: {:?}", response);
                    return Ok("<max tokens reached>".to_owned());
                }
                None => {
                    panic!("No stop reason was provided in {:?}", response);
                }
            };
        }

        Ok("<unexpected end of conversation>".to_owned())
    }

    async fn invoke_tool(&self, response: ClaudeResponse) -> Result<serde_json::Value, Error> {
        let (id, name, params) = response
            .content
            .iter()
            .find_map(|c| match c {
                Content::ToolUse { id, name, input } => Some((id, name, input)),
                _ => None,
            })
            .expect("a tool is provided for execution");

        println!("Tool invocation {} on {}", id, name);
        println!("Parameters: {:?}", params);

        let _ = self
            .tools
            .iter()
            .find_map(|t| if t.0.name == name { Some(t.1) } else { None })
            .expect("a toolkit should exist");

        todo!("this")
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
