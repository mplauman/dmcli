use reqwest::Response;

use crate::errors::Error;
use crate::tools::Toolkit;

pub struct Client {
    pub model: String,
    pub endpoint: String,
    pub client: reqwest::Client,
    pub toolkits: Vec<Box<dyn Toolkit>>,
}

impl Client {
    pub async fn request<T: serde::Serialize + ?Sized>(
        &self,
        messages: &T,
    ) -> Result<Response, Error> {
        let tools: Vec<serde_json::Value> = self
            .toolkits
            .iter()
            .flat_map(|tk| tk.list_tools())
            .collect();

        let request = self
            .client
            .post(self.endpoint.as_str())
            .json(&serde_json::json!({
                "model": self.model.as_str(),
                "max_tokens": 1024,
                "tools": tools,
                "messages": messages,
            }))
            .build()?;

        println!("Request: {:?}", request);

        let body_bytes = request.body().unwrap().as_bytes().unwrap();
        let text = String::from_utf8(body_bytes.to_vec()).unwrap();

        println!("Request body: {}", text);

        let response = self.client.execute(request).await?;
        println!("Response: {:?}", response);

        Ok(response)
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

        let client = Client {
            model: self.model,
            endpoint: self.endpoint,
            toolkits: self.toolkits,
            client: reqwest::ClientBuilder::default()
                .default_headers(headers)
                .build()?,
        };

        Ok(client)
    }
}
