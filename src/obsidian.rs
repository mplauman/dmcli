use std::path::PathBuf;

use crate::tools::{Tool, Toolkit};

pub struct LocalVault {
    vault: PathBuf,
}

impl LocalVault {
    #[allow(dead_code)] // TODO
    pub async fn list_vault_contents(&self) -> Vec<String> {
        let entries = std::fs::read_dir(&self.vault).unwrap();
        let mut files = Vec::new();

        for entry in entries {
            let Ok(entry) = entry else {
                continue;
            };
            let Ok(file_name) = entry.file_name().into_string() else {
                continue;
            };

            files.push(file_name);
        }

        files
    }
}

impl Toolkit for LocalVault {
    fn list_tools(&self) -> Vec<Tool> {
        vec![Tool {
            name: "get_session_notes",
            description: "Returns a list of the dungeon master's local notes",
            input_schema: serde_json::json!({
                "type": "object",
                "parameters": {},
                "required": [],
            }),
        }]
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
