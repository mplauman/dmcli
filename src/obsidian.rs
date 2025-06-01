use std::{collections::VecDeque, path::PathBuf};

use rmcp::{
    ServerHandler,
    model::{CallToolResult, Content},
    schemars, tool,
};

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct GrepRequest {
    #[schemars(description = "the pattern to search for.")]
    pub query: String,

    #[schemars(
        description = "the location to search. it can be either a directory or a filename."
    )]
    pub search_path: String,

    #[schemars(description = "whether to search recursively or not.")]
    pub recursive: bool,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct ListFilesRequest {
    #[schemars(
        description = "the directory to start listing from. if unset or blank the entire vault contents will be returned"
    )]
    pub directory: Option<String>,

    #[schemars(description = "whether to search recursively or not. defaults to 'false' if unset")]
    pub recursive: Option<bool>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct ReadTextFileRequest {
    #[schemars(description = "the fully qualified name of the file to read")]
    pub filename: String,
}

#[derive(Clone)]
pub struct Obsidian {
    vault: PathBuf,
}

#[tool(tool_box)]
impl Obsidian {
    pub fn new(vault: PathBuf) -> Self {
        Self { vault }
    }

    #[tool(
        description = "list all the files in a directory. returned directories will end in a trailin"
    )]
    pub fn list_vaule(
        &self,
        #[tool(aggr)] ListFilesRequest {
            directory,
            recursive,
        }: ListFilesRequest,
    ) -> Result<CallToolResult, rmcp::Error> {
        let mut to_scan = VecDeque::<PathBuf>::new();
        to_scan.push_back(directory.map(PathBuf::from).unwrap_or(self.vault.clone()));

        let mut files = Vec::<String>::new();
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
                    if recursive.unwrap_or(false) {
                        to_scan.push_back(entry_path.clone());
                    }

                    files.push(path_string + "/");
                } else {
                    files.push(path_string);
                }
            }
        }

        let result = serde_json::json!(files);

        Ok(CallToolResult::success(vec![Content::json(result)?]))
    }

    #[tool(
        description = "search through files and folders for a pattern. returns a list of files that include the requested pattern."
    )]
    pub fn grep(
        &self,
        #[tool(aggr)] GrepRequest {
            query,
            search_path,
            recursive,
        }: GrepRequest,
    ) -> Result<CallToolResult, rmcp::Error> {
        log::info!(
            "Searching for {} in {} (recursive: {})",
            query,
            search_path,
            recursive
        );

        let mut to_scan = VecDeque::<PathBuf>::new();
        to_scan.push_back(self.vault.join(search_path));

        let mut matching_files = Vec::<String>::new();

        while let Some(path) = to_scan.pop_front() {
            if path.is_file() {
                let Ok(contents) = std::fs::read_to_string(&path) else {
                    log::info!("Failed to read text from {}, skipping", path.display());
                    continue;
                };

                if !contents.contains(&query) {
                    continue;
                }

                let Ok(relative_path) = path.strip_prefix(&self.vault) else {
                    log::info!("Failed to extract relative path from {}", path.display());
                    continue;
                };

                matching_files.push(relative_path.to_str().unwrap().to_owned());
                continue;
            }

            if !recursive {
                continue;
            }

            if !path.is_dir() {
                log::warn!("Skipping non-directory: {}", path.display());
                continue;
            }

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

                to_scan.push_back(entry.path());
            }
        }

        log::info!(
            "Found matching files: {}",
            serde_json::to_string(&matching_files).unwrap()
        );
        let files = vec![Content::json(serde_json::json!(matching_files))?];

        let result = CallToolResult::success(files);

        Ok(result)
    }

    #[tool(
        description = "read the contents of a text file. this should only be used for files that contain text, such as markdown (.md) or text (.txt)"
    )]
    pub fn read_text_file(
        &self,
        #[tool(aggr)] ReadTextFileRequest { filename }: ReadTextFileRequest,
    ) -> Result<CallToolResult, rmcp::Error> {
        let path = self.vault.join(filename);

        let contents = std::fs::read_to_string(path).expect("failed to read file");
        let result = CallToolResult::success(vec![Content::text(contents)]);

        Ok(result)
    }
}

#[tool(tool_box)]
impl ServerHandler for Obsidian {}
