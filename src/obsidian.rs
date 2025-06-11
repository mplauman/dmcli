use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::UNIX_EPOCH;

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

/// Request parameters for the get_vault_structure function
#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct GetVaultStructureRequest {
    /// Optional folder path to get structure of a subsection
    /// If not provided, returns the entire vault structure
    #[schemars(
        description = "Optional folder path to get structure of a subsection. If not provided, returns the entire vault structure."
    )]
    pub folder_path: Option<String>,
}

/// Represents the hierarchical structure of a directory
///
/// This struct is used to build a tree-like representation of the
/// directory structure, with information about files and subdirectories
/// at each level.
#[derive(serde::Serialize)]
pub struct DirectoryInfo {
    /// Name of the current directory
    pub name: String,
    /// Total number of files in this directory and all subdirectories
    pub file_count: usize,
    /// List of files directly in this directory
    pub files: Vec<String>,
    /// Map of subdirectory names to their directory information
    pub subdirectories: HashMap<String, DirectoryInfo>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct ReadTextFileRequest {
    #[schemars(
        description = "the fully qualified name of the file to read. this should *only* be a file that exists in the vault. do not guess filenames."
    )]
    pub filename: String,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct GetFileMetadataRequest {
    #[schemars(
        description = "the filename to extract metadata from. this should be a markdown file in the vault."
    )]
    pub filename: String,
}

#[derive(serde::Serialize)]
pub struct FileMetadata {
    /// Creation date of the file (timestamp)
    pub creation_date: u64,
    /// Modification date of the file (timestamp)
    pub modification_date: u64,
    /// Tags extracted from the markdown content (format: #tag)
    pub tags: Vec<String>,
    /// Links extracted from the markdown content (format: [[link]])
    pub links: Vec<String>,
    /// Frontmatter properties as key-value pairs from Markdown frontmatter (--- delimited section)
    pub frontmatter: HashMap<String, String>,
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

    /// Build a hierarchical representation of the directory structure
    ///
    /// This function recursively traverses the directory structure starting from the given
    /// base path and constructs a tree-like representation with file counts at each level.
    ///
    /// # Arguments
    ///
    /// * `base_path` - The starting directory path to build the structure from
    ///
    /// # Returns
    ///
    /// A `DirectoryInfo` struct containing the hierarchical directory structure
    fn build_directory_structure(&self, base_path: &PathBuf) -> DirectoryInfo {
        let walk = ignore::WalkBuilder::new(base_path)
            .hidden(false)
            .standard_filters(true)
            .follow_links(true)
            .build();

        let mut root_name = base_path
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| "vault".to_string());

        // If we're building from the vault root, use a specific name
        if base_path == &self.vault {
            root_name = "vault".to_string();
        }

        let mut dir_info = DirectoryInfo {
            name: root_name,
            file_count: 0,
            files: Vec::new(),
            subdirectories: HashMap::new(),
        };

        // First pass: collect all entries
        let mut entries = Vec::new();
        for result in walk {
            let Ok(entry) = result else {
                log::warn!("Failed to read {:?}", result);
                continue;
            };
            entries.push(entry);
        }

        // Process files first
        for entry in &entries {
            let Some(file_type) = entry.file_type() else {
                log::warn!("Failed to get file type from {:?}", entry);
                continue;
            };

            let path = entry.path();

            if !file_type.is_dir() {
                // Skip files not directly in this directory
                if path.parent().map(|p| p != base_path).unwrap_or(true) {
                    continue;
                }

                let filename = path
                    .file_name()
                    .map(|name| name.to_string_lossy().to_string())
                    .unwrap_or_default();

                dir_info.files.push(filename);
                dir_info.file_count += 1;
            }
        }

        // Process subdirectories
        for entry in &entries {
            let Some(file_type) = entry.file_type() else {
                continue;
            };

            let path = entry.path();

            if file_type.is_dir() && path != base_path {
                // Skip nested directories that aren't direct children
                if path.parent().map(|p| p != base_path).unwrap_or(true) {
                    continue;
                }

                let subdir_name = path
                    .file_name()
                    .map(|name| name.to_string_lossy().to_string())
                    .unwrap_or_default();

                let subdir_info = self.build_directory_structure(&path.to_path_buf());
                dir_info.file_count += subdir_info.file_count;
                dir_info.subdirectories.insert(subdir_name, subdir_info);
            }
        }

        dir_info
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

    /// Returns a hierarchical representation of the vault's folder structure
    ///
    /// This function helps an LLM understand how notes are organized within the vault by
    /// providing a comprehensive view of the directory structure, including file counts
    /// and hierarchical relationships between directories.
    ///
    /// The returned structure is a tree-like JSON representation that can be traversed
    /// to understand the organization of notes, which is especially helpful for
    /// discovering the organizational system used by the DM.
    #[tool(
        description = "returns a hierarchical representation of the vault's folder structure. helps understand how the notes are organized. returns JSON with directory structure and file counts."
    )]
    pub fn get_vault_structure(
        &self,
        #[tool(aggr)] GetVaultStructureRequest { folder_path }: GetVaultStructureRequest,
    ) -> Result<CallToolResult, rmcp::Error> {
        let base_path = if let Some(folder) = folder_path {
            let path = std::path::Path::new(&folder);
            if path.is_absolute() {
                log::warn!("Absolute path provided: {}", folder);
                self.vault.clone()
            } else {
                let combined_path = self.vault.join(path);
                if combined_path.exists() && combined_path.is_dir() {
                    combined_path
                } else {
                    log::warn!(
                        "Folder path does not exist or is not a directory: {}",
                        folder
                    );
                    self.vault.clone()
                }
            }
        } else {
            self.vault.clone()
        };

        log::info!("Building directory structure for {}", base_path.display());
        let structure = self.build_directory_structure(&base_path);

        let result = serde_json::json!(structure);
        log::info!("Directory structure built successfully");

        Ok(CallToolResult::success(vec![Content::json(result)?]))
    }

    /// Extracts metadata from markdown files
    ///
    /// This function analyzes a markdown file and extracts useful metadata including:
    /// - File creation and modification dates
    /// - Tags (formatted as #tag in the content)
    /// - Links to other notes (formatted as [[link]] in the content)
    /// - Frontmatter properties from the markdown frontmatter
    ///
    /// The frontmatter is expected to be at the beginning of the file,
    /// enclosed between `---` lines, which is a common format in Markdown files.
    #[tool(
        description = "extracts metadata from markdown files. returns creation date, modification date, tags, links, and frontmatter properties."
    )]
    pub fn get_file_metadata(
        &self,
        #[tool(aggr)] GetFileMetadataRequest { filename }: GetFileMetadataRequest, // filename must be a Markdown file
    ) -> Result<CallToolResult, rmcp::Error> {
        let filename_copy = filename.clone();
        let path = std::path::Path::new(&filename);
        let full_path = if path.is_absolute() {
            PathBuf::from(filename)
        } else {
            self.vault.join(path)
        };

        if !full_path.exists() || !full_path.is_file() {
            log::warn!("File does not exist or is not a file: {}", filename_copy);
            panic!("File not found: {}", filename_copy);
        }

        // Get file metadata from filesystem
        let metadata = fs::metadata(&full_path).expect("failed to read file metadata");

        // Extract creation and modification times
        let creation_date = metadata
            .created()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|duration| duration.as_secs())
            .unwrap_or(0);

        let modification_date = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|duration| duration.as_secs())
            .unwrap_or(0);

        // Read file content
        let content = std::fs::read_to_string(&full_path).expect("failed to read file content");

        // Extract frontmatter from Markdown files (between --- delimiters)
        let mut frontmatter = HashMap::new();
        let content_without_frontmatter = if let Some(stripped) = content.strip_prefix("---") {
            if let Some(end_index) = stripped.find("---") {
                let fm_content = &stripped[0..end_index];

                // Simple key-value parsing for Markdown frontmatter
                for line in fm_content.lines() {
                    if let Some(idx) = line.find(':') {
                        let key = line[0..idx].trim().to_string();
                        let value = line[idx + 1..].trim().to_string();
                        if !key.is_empty() {
                            frontmatter.insert(key, value);
                        }
                    }
                }

                // Return content after frontmatter
                &stripped[end_index + 3..]
            } else {
                &content
            }
        } else {
            &content
        };

        // Extract tags (#tag)
        let mut tags = Vec::new();
        for word in content_without_frontmatter.split_whitespace() {
            if word.starts_with('#') && word.len() > 1 {
                let tag = word[1..]
                    .trim_end_matches(|c: char| !c.is_alphanumeric())
                    .to_string();
                if !tag.is_empty() && !tags.contains(&tag) {
                    tags.push(tag);
                }
            }
        }

        // Extract links ([[link]])
        let mut links = Vec::new();
        let mut current_pos = 0;
        while let Some(start_idx) = content_without_frontmatter[current_pos..].find("[[") {
            current_pos += start_idx + 2;
            if let Some(end_idx) = content_without_frontmatter[current_pos..].find("]]") {
                let link = content_without_frontmatter[current_pos..current_pos + end_idx]
                    .trim()
                    .to_string();
                if !link.is_empty() && !links.contains(&link) {
                    links.push(link);
                }
                current_pos += end_idx + 2;
            } else {
                break;
            }
        }

        // Create metadata object
        let file_metadata = FileMetadata {
            creation_date,
            modification_date,
            tags,
            links,
            frontmatter,
        };

        let result = serde_json::json!(file_metadata);
        log::info!("File metadata extracted successfully");

        Ok(CallToolResult::success(vec![Content::json(result)?]))
    }
}

#[tool(tool_box)]
impl ServerHandler for Obsidian {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::TempDir;

    /// Helper function to create a test vault with a specific structure
    fn create_test_vault() -> TempDir {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");

        // Create a simple vault structure
        // vault/
        // ├── characters/
        // │   ├── npc1.md
        // │   └── npc2.md
        // ├── locations/
        // │   └── city.md
        // └── notes.md

        // Create the base structure
        let char_dir = temp_dir.path().join("characters");
        let loc_dir = temp_dir.path().join("locations");
        fs::create_dir(&char_dir).expect("Failed to create characters directory");
        fs::create_dir(&loc_dir).expect("Failed to create locations directory");

        // Create some files with different formats
        let files = [
            (
                char_dir.join("npc1.md"),
                "# NPC 1\nThis is a test NPC with a #character tag and a link to [[city]]",
            ),
            (char_dir.join("npc2.md"), "# NPC 2\nAnother test NPC"),
            (loc_dir.join("city.md"), "# Test City\nA fictional place"),
            (
                temp_dir.path().join("notes.md"),
                "# General Notes\nCampaign notes go here",
            ),
            (
                temp_dir.path().join("with_frontmatter.md"),
                "---\ntitle: Frontmatter Test\ntype: note\ntags: important, reference\n---\n# Document with Frontmatter\nThis document has YAML frontmatter and #metadata tags with a [[link_test]]",
            ),
        ];

        for (path, content) in &files {
            let mut file = fs::File::create(path).expect("Failed to create file");
            file.write_all(content.as_bytes())
                .expect("Failed to write to file");
        }

        temp_dir
    }

    #[test]
    fn test_get_vault_structure_root() {
        // Create a test vault
        let temp_dir = create_test_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Get the structure of the entire vault
        let structure = obsidian.build_directory_structure(&temp_dir.path().to_path_buf());

        // Validate the structure
        assert_eq!(structure.name, "vault");
        assert_eq!(structure.file_count, 4); // Total of 4 files in the vault
        assert_eq!(structure.files.len(), 1); // 1 file directly in the root
        assert_eq!(structure.subdirectories.len(), 2); // 2 subdirectories

        // Validate the characters directory
        let chars = structure
            .subdirectories
            .get("characters")
            .expect("Characters directory not found");
        assert_eq!(chars.name, "characters");
        assert_eq!(chars.file_count, 2); // 2 character files
        assert_eq!(chars.files.len(), 2); // Both files directly in this directory
        assert_eq!(chars.subdirectories.len(), 0); // No subdirectories

        // Validate the locations directory
        let locs = structure
            .subdirectories
            .get("locations")
            .expect("Locations directory not found");
        assert_eq!(locs.name, "locations");
        assert_eq!(locs.file_count, 1); // 1 location file
        assert_eq!(locs.files.len(), 1); // File directly in this directory
        assert_eq!(locs.subdirectories.len(), 0); // No subdirectories
    }

    #[test]
    fn test_get_vault_structure_subfolder() {
        // Create a test vault
        let temp_dir = create_test_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Get the structure of just the characters subfolder
        let char_path = temp_dir.path().join("characters");
        let structure = obsidian.build_directory_structure(&char_path);

        // Validate the structure
        assert_eq!(structure.name, "characters");
        assert_eq!(structure.file_count, 2); // 2 files in this directory
        assert_eq!(structure.files.len(), 2); // Both files directly in this directory
        assert_eq!(structure.subdirectories.len(), 0); // No subdirectories

        // Verify file names
        assert!(structure.files.contains(&"npc1.md".to_string()));
        assert!(structure.files.contains(&"npc2.md".to_string()));
    }

    #[test]
    fn test_get_vault_structure_tool() {
        // Create a test vault
        let temp_dir = create_test_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Test the actual tool function
        let request = GetVaultStructureRequest { folder_path: None };
        let result = obsidian
            .get_vault_structure(request)
            .expect("Tool function failed");

        // Verify that we got a successful result with JSON content
        let content = result.content;
        assert_eq!(content.len(), 1); // Should have exactly one content item

        // We can't easily deserialize the JSON directly, but we can verify it contains expected data
        let content_str = format!("{:?}", content[0]);
        assert!(content_str.contains("vault"));
        assert!(content_str.contains("characters"));
        assert!(content_str.contains("locations"));
        assert!(content_str.contains("npc1.md"));
    }

    #[test]
    fn test_get_vault_structure_with_subfolder() {
        // Create a test vault
        let temp_dir = create_test_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Test with a specific subfolder
        let request = GetVaultStructureRequest {
            folder_path: Some("characters".to_string()),
        };
        let result = obsidian
            .get_vault_structure(request)
            .expect("Tool function failed");

        // Verify that we got a successful result with only characters data
        let content = result.content;
        let content_str = format!("{:?}", content[0]);
        assert!(content_str.contains("characters"));
        assert!(content_str.contains("npc1.md"));
        assert!(content_str.contains("npc2.md"));
        assert!(!content_str.contains("city.md")); // Should not include location files
    }

    #[test]
    fn test_get_vault_structure_invalid_folder() {
        // Create a test vault
        let temp_dir = create_test_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Test with a non-existent folder
        let request = GetVaultStructureRequest {
            folder_path: Some("non_existent_folder".to_string()),
        };
        let result = obsidian
            .get_vault_structure(request)
            .expect("Tool function failed");

        // Should default to the root vault
        let content = result.content;
        let content_str = format!("{:?}", content[0]);
        assert!(content_str.contains("vault")); // Should default to vault root
        assert!(content_str.contains("characters"));
        assert!(content_str.contains("locations"));
    }

    #[test]
    fn test_get_file_metadata_with_tags_and_links() {
        // Create a test vault
        let temp_dir = create_test_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Test file with tags and links
        let file_path = temp_dir
            .path()
            .join("characters")
            .join("npc1.md")
            .to_string_lossy()
            .to_string();
        let request = GetFileMetadataRequest {
            filename: file_path,
        };

        let result = obsidian
            .get_file_metadata(request)
            .expect("Tool function failed");

        // Verify the result
        let content = result.content;
        let content_str = format!("{:?}", content[0]);

        // Should contain tag and link
        assert!(content_str.contains("character")); // The tag without the # symbol
        assert!(content_str.contains("city")); // The link without the [[ ]] symbols
    }

    #[test]
    fn test_get_file_metadata_with_frontmatter() {
        // Create a test vault
        let temp_dir = create_test_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Test file with Markdown frontmatter
        let file_path = temp_dir
            .path()
            .join("with_frontmatter.md")
            .to_string_lossy()
            .to_string();
        let request = GetFileMetadataRequest {
            filename: file_path,
        };

        let result = obsidian
            .get_file_metadata(request)
            .expect("Tool function failed");

        // Verify the result
        let content = result.content;
        let content_str = format!("{:?}", content[0]);

        // Should contain frontmatter properties, tag, and link
        assert!(content_str.contains("title"));
        assert!(content_str.contains("Frontmatter Test"));
        assert!(content_str.contains("type"));
        assert!(content_str.contains("note"));
        assert!(content_str.contains("metadata")); // The tag without the # symbol
        assert!(content_str.contains("link_test")); // The link without the [[ ]] symbols
    }

    #[test]
    #[should_panic]
    fn test_get_file_metadata_invalid_file() {
        // Create a test vault
        let temp_dir = create_test_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Test with a non-existent markdown file
        let request = GetFileMetadataRequest {
            filename: "non_existent_file.md".to_string(),
        };

        // Should panic with "File not found"
        let _ = obsidian.get_file_metadata(request);
    }
}
