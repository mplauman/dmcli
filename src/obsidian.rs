use std::{collections::VecDeque, path::PathBuf};

use crate::tools::{Tool, Toolkit};

pub struct LocalVault {
    vault: PathBuf,
}

impl LocalVault {
    #[allow(dead_code)] // TODO
    pub fn list_vault_contents(&self) -> serde_json::Value {
        let mut to_scan = VecDeque::<PathBuf>::new();
        to_scan.push_back(self.vault.clone());

        let mut files = Vec::new();
        while let Some(path) = to_scan.pop_front() {
            let entries = std::fs::read_dir(&path).unwrap();

            for entry in entries {
                let Ok(entry) = entry else {
                    continue;
                };
                if entry
                    .file_name()
                    .to_str()
                    .map(|it| it.starts_with("."))
                    .unwrap_or(false)
                {
                    continue;
                }

                let entry_path = entry.path();
                let path_string = entry_path.to_str().unwrap().to_owned();

                if entry_path.is_dir() {
                    to_scan.push_back(entry_path.clone());
                    continue;
                }

                files.push(path_string);
            }
        }

        serde_json::json!(files)
    }

    fn read_markdown_file(&self, params: &serde_json::Value) -> serde_json::Value {
        let path = params["path"].as_str().expect("path is a string");
        let path = self.vault.join(path);
        let contents = std::fs::read_to_string(path).expect("failed to read file");

        serde_json::json!({
            "contents": contents,
        })
    }
}

impl Toolkit for LocalVault {
    fn list_tools(&self) -> Vec<Tool> {
        vec![
            Tool {
                name: "get_session_notes",
                description: "Returns a list of the dungeon master's local notes",
                input_schema: serde_json::json!({
                    "type": "object",
                    "parameters": {},
                    "required": [],
                }),
            },
            Tool {
                name: "read_markdown_file",
                description: "Return the contents of a markdown file",
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the markdown file",
                        },
                    },
                    "required": ["path"],
                    "additionalProperties": false,
                }),
            },
        ]
    }

    fn run_tool(&self, name: &str, params: &serde_json::Value) -> serde_json::Value {
        match name {
            "get_session_notes" => self.list_vault_contents(),
            "read_markdown_file" => self.read_markdown_file(params),
            _ => panic!("don't know how to handle {}", name),
        }
    }
}

#[derive(Default)]
pub struct LocalVaultBuilder {
    directory: Option<PathBuf>,
}

impl LocalVaultBuilder {
    pub fn with_directory(self, directory: PathBuf) -> Self {
        let directory = Some(directory);

        Self { directory }
    }

    pub fn build(self) -> LocalVault {
        LocalVault {
            vault: self.directory.expect("the vault directory has been set"),
        }
    }
}
