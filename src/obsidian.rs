use std::path::PathBuf;

use rmcp::{
    ServerHandler,
    model::{CallToolResult, Content},
    schemars, tool,
};

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct GrepRequest {
    #[schemars(
        description = "a case insensitive pattern to search for. this IS NOT a regular expression. provide only the
       exact string to search for."
    )]
    pub query: String,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct ListFilesRequest {}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct ReadTextFileRequest {
    #[schemars(
        description = "the fully qualified name of the file to read. this should *only* be a file that exists in the vault. do not guess filenames."
    )]
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

    /// Recursively find a list of files in a directory. If a directory is not provided then
    /// the entire vault will be listed.
    fn internal_list_files(&self) -> Vec<PathBuf> {
        let walk = ignore::WalkBuilder::new(&self.vault)
            .hidden(false)
            .standard_filters(true)
            .follow_links(true)
            .build();

        let mut files = Vec::<PathBuf>::new();
        for result in walk {
            let Ok(entry) = result else {
                log::warn!("Failed to read {:?}", result);
                continue;
            };

            let Some(file_type) = entry.file_type() else {
                log::warn!("Failed to get file type from {:?}", entry);
                continue;
            };

            if !file_type.is_dir() {
                let path = entry.path();
                files.push(path.into());
            }
        }

        files
    }

    #[tool(
        description = "recursively lists all files in a directory. if provided a path, this will list all files
        within that path recursively. if no path is provided, this will list all files starting from the vault
        root. the returned values are absolute paths to the matching files."
    )]
    pub fn list_files(
        &self,
        #[tool(aggr)] ListFilesRequest {}: ListFilesRequest,
    ) -> Result<CallToolResult, rmcp::Error> {
        let files = self.internal_list_files();
        let result = serde_json::json!(files);

        Ok(CallToolResult::success(vec![Content::json(result)?]))
    }

    #[tool(
        description = "search through files and folders for a pattern. returns a
        list of files that include the requested pattern."
    )]
    pub fn grep(
        &self,
        #[tool(aggr)] GrepRequest { query }: GrepRequest,
    ) -> Result<CallToolResult, rmcp::Error> {
        use grep::{
            regex::RegexMatcherBuilder,
            searcher::{Encoding, SearcherBuilder, sinks::Bytes},
        };

        log::info!("Searching for {}", query);

        let matcher = RegexMatcherBuilder::new()
            .case_insensitive(true)
            .fixed_strings(true)
            .build(&query)
            .expect("the regex will work");

        let mut searcher = SearcherBuilder::new()
            .line_number(true)
            .encoding(Some(Encoding::new("UTF-8").unwrap()))
            .build();

        let mut matching_files = Vec::<String>::new();

        // if this is a directory then find all the files. otherwise just read the file
        for path in self.internal_list_files() {
            let sink = Bytes(|lnum, _bytes| {
                log::info!("Found match in {}, line {}", path.display(), lnum);

                matching_files.push(path.to_string_lossy().to_string());
                Ok(false)
            });

            searcher
                .search_path(&matcher, &path, sink)
                .expect("the search works");
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
        description = "read the contents of a text file. this should only be used for files that
        contain text, such as markdown (.md) or text (.txt)"
    )]
    pub fn read_text_file(
        &self,
        #[tool(aggr)] ReadTextFileRequest { filename }: ReadTextFileRequest,
    ) -> Result<CallToolResult, rmcp::Error> {
        let contents = std::fs::read_to_string(filename).expect("failed to read file");
        let result = CallToolResult::success(vec![Content::text(contents)]);

        Ok(result)
    }
}

#[tool(tool_box)]
impl ServerHandler for Obsidian {}
