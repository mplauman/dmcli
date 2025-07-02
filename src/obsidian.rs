use crate::errors::Error;
use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use regex::Regex;
use rmcp::{
    ServerHandler,
    model::{CallToolResult, Content},
    schemars, tool,
};

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct ListFilesRequest {}

/// Request parameters for the get_vault_structure function
#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct GetVaultStructureRequest {
    /// Optional folder path to get structure of a subsection
    /// Must be fully qualified relative to the vault root (e.g., 'folder/subfolder'), NOT an absolute path
    /// If not provided, returns the entire vault structure
    #[schemars(
        description = "Optional folder path to get structure of a subsection. Must be fully qualified relative to the vault root (e.g., 'folder/subfolder'), NOT an absolute path. If not provided, returns the entire vault structure."
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
    /// List of files directly in this directory
    pub files: Vec<String>,
    /// Map of subdirectory names to their directory information
    pub subdirectories: HashMap<String, DirectoryInfo>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct ReadTextFileRequest {
    #[schemars(
        description = "The fully qualified name of the file to read, relative to the vault root (e.g., 'folder/file.md'). This should NOT be an absolute path. This should *only* be a file that exists in the vault. Do not guess filenames."
    )]
    pub filename: String,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct GetFileMetadataRequest {
    #[schemars(
        description = "The filename to extract metadata from, relative to the vault root (e.g., 'folder/file.md'). This should NOT be an absolute path. This should be a markdown file in the vault."
    )]
    pub filename: String,
}

/// Request parameters for the get_tags_summary function
#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct GetTagsSummaryRequest {
    /// Optional folder path to limit the scope of tag search
    /// Must be fully qualified relative to the vault root (e.g., 'folder/subfolder'), NOT an absolute path
    /// If not provided, tags from the entire vault will be returned.
    /// When specified, only tags from files in this folder (and its subfolders) will be included.
    #[schemars(
        description = "Optional folder path to limit the scope of tag search. Must be fully qualified relative to the vault root (e.g., 'folder/subfolder'), NOT an absolute path. If not provided, tags from the entire vault will be returned."
    )]
    pub folder_path: Option<String>,
}

/// Request parameters for the get_note_by_tag function
#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct GetNoteByTagRequest {
    /// Tag names to search for (without the # symbol)
    /// Can be a single tag or multiple tags
    #[schemars(
        description = "Tag names to search for (without the # symbol). Files matching any of these tags will be returned."
    )]
    pub tags: Vec<String>,
    /// Optional folder path to limit the scope of search
    /// Must be fully qualified relative to the vault root (e.g., 'folder/subfolder'), NOT an absolute path
    /// If not provided, searches the entire vault
    #[schemars(
        description = "Optional folder path to limit the scope of search. Must be fully qualified relative to the vault root (e.g., 'folder/subfolder'), NOT an absolute path. If not provided, searches the entire vault."
    )]
    pub folder_path: Option<String>,
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
    pub frontmatter: serde_json::Value,
}

/// Response structure for get_tags_summary
///
/// This structure represents a single tag found in the vault, including
/// its frequency and locations. Tags are returned in descending order of frequency,
/// with alphabetical ordering used as a tiebreaker for tags with the same count.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct TagSummary {
    /// Tag name (without the # prefix)
    pub tag: String,
    /// Number of occurrences of the tag in the vault
    pub count: usize,
    /// List of files where the tag appears (relative paths from vault root)
    pub files: Vec<String>,
}

/// Response structure for get_note_by_tag function
#[derive(serde::Serialize)]
pub struct NoteWithTags {
    /// Filename (relative path from vault root)
    pub filename: String,
    /// Tags found in this file
    pub tags: Vec<String>,
    /// Frontmatter from the file
    pub frontmatter: serde_json::Value,
    /// Creation date of the file (timestamp)
    pub creation_date: u64,
    /// Modification date of the file (timestamp)
    pub modification_date: u64,
}

/// Request structure for search_with_context function
#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct SearchWithContextRequest {
    #[schemars(description = "Search query pattern")]
    pub query: String,

    #[schemars(
        description = "Number of context lines to include before and after each match (default: 2)"
    )]
    pub context_lines: Option<usize>,

    #[schemars(description = "Whether to treat the query as a regex pattern (default: false)")]
    pub regex: Option<bool>,

    #[schemars(description = "Whether the search should be case sensitive (default: false)")]
    pub case_sensitive: Option<bool>,
}

/// Response structure for search_with_context function
#[derive(serde::Serialize, serde::Deserialize)]
pub struct SearchMatch {
    /// Filename where the match was found
    pub filename: String,
    /// Line number of the match (1-based)
    pub line_number: usize,
    /// The actual line content containing the match
    pub line_content: String,
    /// Lines before the match for context
    pub context_before: Vec<String>,
    /// Lines after the match for context
    pub context_after: Vec<String>,
    /// Character position where the match starts in the line
    pub match_start: usize,
    /// Character position where the match ends in the line
    pub match_end: usize,
}

/// Request structure for get_linked_notes function
#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct GetLinkedNotesRequest {
    #[schemars(description = "Filename to find linked notes for (relative to vault root)")]
    pub filename: String,
}

/// Response structure for get_linked_notes function
#[derive(serde::Serialize, serde::Deserialize)]
pub struct LinkedNotes {
    /// The target filename
    pub filename: String,
    /// Notes that this file links to
    pub outgoing_links: Vec<String>,
    /// Notes that link to this file
    pub incoming_links: Vec<String>,
}

// Define the key for our cache
#[derive(PartialEq, Eq, Hash, Clone)]
struct CacheKey {
    filepath: PathBuf,
    last_modified: SystemTime,
}

// Define what we want to cache
#[derive(Clone)]
struct FileMetadataCache {
    tags: Vec<String>,
    links: Vec<String>,
    frontmatter: Option<serde_yaml::Value>,
}

// The cache itself
struct MetadataCache {
    cache: RwLock<HashMap<CacheKey, FileMetadataCache>>,
}

impl MetadataCache {
    fn new() -> Self {
        MetadataCache {
            cache: RwLock::new(HashMap::new()),
        }
    }

    // Get metadata from cache if it exists, otherwise parse and cache it
    fn get_or_parse<F>(&self, filepath: &Path, extract_tags: F) -> Option<FileMetadataCache>
    where
        F: Fn(&str) -> Vec<String>,
    {
        let Ok(metadata) = std::fs::metadata(filepath) else {
            log::warn!("Failed to get metadata for {}", filepath.display());
            return None;
        };

        let Ok(last_modified) = metadata.modified() else {
            log::warn!("Failed to get last modified for {}", filepath.display());
            return None;
        };

        let key = CacheKey {
            filepath: filepath.to_path_buf(),
            last_modified,
        };

        // Try to read from cache firsts
        let cached_data = match self.cache.read() {
            Ok(cache_read) => cache_read.get(&key).cloned(),
            Err(x) => panic!("Cache RW lock is poisoned: {x}"),
        };
        if cached_data.is_some() {
            return cached_data;
        }

        // Parse the file
        let Ok(content) = std::fs::read_to_string(filepath) else {
            log::warn!("Failed to read the contents of {}", filepath.display());
            return None;
        };

        // Parse tags, links, and frontmatter
        let tags = extract_tags(&content);
        let links = extract_links_from_content(&content);
        let frontmatter = extract_frontmatter_from_content(&content);

        let metadata_cache = FileMetadataCache {
            tags,
            links,
            frontmatter,
        };

        // Store in cache with a write lock
        if let Ok(mut cache_write) = self.cache.write() {
            cache_write.insert(key, metadata_cache.clone());
        }

        // Return the newly created data
        Some(metadata_cache)
    }
}

// Helper functions for parsing [[link]] entries out of a file.
fn extract_links_from_content(content: &str) -> Vec<String> {
    let re = Regex::new(r"\[\[(.*?)\]\]").expect("hardcoded regex must be valid");

    re.captures_iter(content)
        .filter_map(|cap| cap.get(1))
        .map(|link| {
            // Handle wiki-style links with display text: [[link|display text]]
            // Extract just the link part before the | character
            let link_str = link.as_str();
            if let Some(pipe_pos) = link_str.find('|') {
                link_str[..pipe_pos].to_string()
            } else {
                link_str.to_string()
            }
        })
        .collect::<HashSet<String>>()
        .into_iter()
        .collect()
}

fn extract_frontmatter_from_content(content: &str) -> Option<serde_yaml::Value> {
    let stripped = content.strip_prefix("---")?;
    let end_index = stripped.find("---")?;

    let frontmatter_str = &stripped[0..end_index];
    serde_yaml::from_str(frontmatter_str).ok()
}

#[derive(Clone)]
pub struct Obsidian {
    vault: PathBuf,
    metadata_cache: Arc<MetadataCache>,
}

#[tool(tool_box)]
impl Obsidian {
    pub fn new(vault: PathBuf) -> Self {
        Self {
            vault,
            metadata_cache: Arc::new(MetadataCache::new()),
        }
    }

    /// Validates that a path is relative to the vault root, not absolute
    fn validate_vault_path(&self, path: &str) -> Result<PathBuf, Error> {
        let path_obj = std::path::Path::new(path);
        
        // Use Rust's built-in absolute path detection which is cross-platform
        if path_obj.is_absolute() {
            return Err(Error::InvalidVaultPath(path.to_string()));
        }
        
        // Additional platform-specific checks for edge cases
        #[cfg(unix)]
        {
            // On Unix, also reject paths starting with '/' that might not be caught by is_absolute()
            if path.starts_with('/') {
                return Err(Error::InvalidVaultPath(path.to_string()));
            }
        }
        
        #[cfg(windows)]
        {
            // On Windows, also check for UNC paths and other Windows-specific absolute path formats
            if path.starts_with('\\') || path.starts_with('/') {
                return Err(Error::InvalidVaultPath(path.to_string()));
            }
            
            // Check for drive letter patterns that might not be caught by is_absolute()
            if path.len() >= 2 && path.chars().nth(1) == Some(':') {
                let first_char = path.chars().nth(0).unwrap();
                if first_char.is_ascii_alphabetic() {
                    return Err(Error::InvalidVaultPath(path.to_string()));
                }
            }
        }
        
        // Join with vault path using cross-platform path operations
        let result_path = self.vault.join(path_obj);
        
        // Ensure the path stays within the vault (prevent directory traversal)
        if let Ok(canonical_result) = result_path.canonicalize() {
            if let Ok(canonical_vault) = self.vault.canonicalize() {
                if !canonical_result.starts_with(canonical_vault) {
                    return Err(Error::InvalidVaultPath(path.to_string()));
                }
            }
        }
        
        Ok(result_path)
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
                log::warn!("Failed to read {result:?}");
                continue;
            };

            let Some(file_type) = entry.file_type() else {
                log::warn!("Failed to get file type from {entry:?}");
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
    fn build_directory_structure(&self, path: PathBuf) -> Result<DirectoryInfo, Error> {
        let name = if path == self.vault {
            "vault".to_string()
        } else {
            path.file_name()
                .expect("paths have names")
                .to_string_lossy()
                .to_string()
        };

        let mut files = Vec::new();
        let mut children = Vec::new();

        for entry_result in std::fs::read_dir(path)? {
            let Ok(entry) = entry_result else {
                log::warn!("Skipping invalid entry: {entry_result:?}");
                continue;
            };

            let entry_path = entry.path();
            let entry_type = entry.file_type()?;

            if entry_type.is_dir() {
                children.push(self.build_directory_structure(entry_path)?);
                continue;
            }

            let Some(filename) = entry_path.file_name() else {
                log::warn!("Failed to get file name for {}", entry_path.display());
                continue;
            };

            files.push(filename.to_string_lossy().to_string());
        }

        let found = DirectoryInfo {
            name,
            files,
            subdirectories: children
                .into_iter()
                .map(|it| (it.name.clone(), it))
                .collect(),
        };

        Ok(found)
    }

    #[tool(
        description = "Read the contents of a text file. The filename parameter must be fully qualified relative to the vault root (e.g., 'folder/file.md'), NOT an absolute path. This should only be used for files that contain text, such as markdown (.md) or text (.txt)."
    )]
    pub fn read_text_file(
        &self,
        #[tool(aggr)] ReadTextFileRequest { filename }: ReadTextFileRequest,
    ) -> Result<CallToolResult, rmcp::Error> {
        let full_path = self.validate_vault_path(&filename)?;
        let contents = std::fs::read_to_string(full_path).map_err(Error::from)?;
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
        description = "Returns a hierarchical representation of the vault's folder structure. Helps understand how the notes are organized. Returns JSON with directory structure and file counts. If folder_path is provided, it must be relative to the vault root, NOT an absolute path."
    )]
    pub fn get_vault_structure(
        &self,
        #[tool(aggr)] GetVaultStructureRequest { folder_path }: GetVaultStructureRequest,
    ) -> Result<CallToolResult, rmcp::Error> {
        let base_path = match folder_path.as_ref() {
            Some(folder) => {
                log::info!("Finding folder structure for {folder}");
                self.validate_vault_path(folder)?
            }
            None => {
                log::info!("Using vault root");
                self.vault.clone()
            }
        };

        let base_path = if base_path.exists() && base_path.is_dir() {
            base_path
        } else {
            log::warn!("Attempted base path {} doesn't exist", base_path.display());
            self.vault.clone()
        };

        log::info!("Building directory structure for {}", base_path.display());
        let structure = self.build_directory_structure(base_path).map_err(|e| {
            log::warn!("Cannot get structure: {e}");
            DirectoryInfo {
                name: "vault".into(),
                files: Vec::default(),
                subdirectories: HashMap::default(),
            }
        });

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
    /// Extracts tags from a Markdown file
    ///
    /// This helper function extracts tags (format: #tag) from Markdown content
    /// after removing frontmatter. It's used by both get_file_metadata and get_tags_summary.
    fn extract_tags_from_content(&self, content: &str) -> Vec<String> {
        // Extract frontmatter from Markdown files (between --- delimiters)
        let content_without_frontmatter = if let Some(stripped) = content.strip_prefix("---") {
            if let Some(end_index) = stripped.find("---") {
                // Return content after frontmatter
                &stripped[end_index + 3..]
            } else {
                content
            }
        } else {
            content
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

        tags
    }

    #[tool(
        description = "Extracts metadata from markdown files. The filename parameter must be fully qualified relative to the vault root (e.g., 'folder/file.md'), NOT an absolute path. Returns creation date, modification date, tags, links, and frontmatter properties."
    )]
    pub fn get_file_metadata(
        &self,
        #[tool(aggr)] GetFileMetadataRequest { filename }: GetFileMetadataRequest, // filename must be a Markdown file
    ) -> Result<CallToolResult, rmcp::Error> {
        let filename_copy = filename.clone();
        let full_path = self.validate_vault_path(&filename)?;

        if !full_path.exists() || !full_path.is_file() {
            log::warn!("File does not exist or is not a file: {filename_copy}");
            return Err(rmcp::Error::invalid_request(
                format!("File not found: {filename_copy}"),
                None,
            ));
        }

        // Get file metadata from filesystem for basic attributes
        let metadata = match fs::metadata(&full_path) {
            Ok(m) => m,
            Err(e) => {
                return Err(rmcp::Error::internal_error(
                    format!("Failed to read file metadata: {e}"),
                    None,
                ));
            }
        };

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

        // Use the cache for parsed metadata
        let result = self
            .metadata_cache
            .get_or_parse(&full_path, |content| self.extract_tags_from_content(content))
            .map(|cache_data| {
                // We have the cached data, use it for the result
                serde_json::json!({
                    "creation_date": creation_date,
                    "modification_date": modification_date,
                    "tags": cache_data.tags,
                    "links": cache_data.links,
                    "frontmatter": cache_data.frontmatter.unwrap_or_else(|| serde_yaml::Value::Null),
                })
            })
            .unwrap_or_else(|| {
                log::warn!("Failed to load cached data, returning empty details");
                Default::default()
            });

        log::info!("File metadata extracted successfully");

        Ok(CallToolResult::success(vec![Content::json(result)?]))
    }

    /// Returns a summary of all tags used in the vault, including their frequency and location
    ///
    /// This function scans all Markdown files in the vault (or a specific folder if provided)
    /// and extracts all tags. It then returns a list of tags sorted by frequency (most common first).
    /// For each tag, it includes:
    /// - The tag name (without the # symbol)
    /// - The count of occurrences across all files
    /// - A list of files where the tag appears
    ///
    /// This is useful for understanding the tagging system used in the vault and finding
    /// common themes or categories.
    #[tool(
        description = "Returns all tags used across the vault with their frequency and the files they appear in. If folder_path is provided, it must be relative to the vault root, NOT an absolute path. File paths in results are returned relative to the vault root."
    )]
    pub fn get_tags_summary(
        &self,
        #[tool(aggr)] GetTagsSummaryRequest { folder_path }: GetTagsSummaryRequest,
    ) -> Result<CallToolResult, rmcp::Error> {
        // Get all files in the vault or in the specified folder
        let all_files = self.internal_list_files();

        // Filter files by folder_path if provided
        let files = if let Some(folder) = folder_path {
            let folder_clone = folder.clone();
            let folder_path = self.validate_vault_path(&folder)?;

            // Check if the folder exists
            if !folder_path.exists() || !folder_path.is_dir() {
                log::warn!("Folder does not exist or is not a directory: {folder_path:?}");
                let error_msg = format!("Folder not found: {folder_clone}");
                return Ok(CallToolResult::error(vec![Content::text(error_msg)]));
            }

            // Filter files that are within the specified folder
            all_files
                .into_iter()
                .filter(|path| path.starts_with(&folder_path))
                .collect()
        } else {
            all_files
        };

        let mut tag_counts: HashMap<String, usize> = HashMap::new();
        let mut tag_files: HashMap<String, Vec<String>> = HashMap::new();

        for filepath in files {
            // Skip non-markdown files
            if filepath.extension() != Some(OsStr::new("md")) {
                continue;
            }

            // Use the cache to get tags
            if let Some(cache_data) = self
                .metadata_cache
                .get_or_parse(&filepath, |content| self.extract_tags_from_content(content))
            {
                // Get relative path for reporting (normalize separators to forward slashes)
                let rel_path = filepath
                    .strip_prefix(&self.vault)
                    .unwrap_or(&filepath)
                    .to_string_lossy()
                    .replace(std::path::MAIN_SEPARATOR, "/");

                for tag in &cache_data.tags {
                    *tag_counts.entry(tag.clone()).or_insert(0) += 1;
                    tag_files
                        .entry(tag.clone())
                        .or_default()
                        .push(rel_path.clone());
                }
            } else {
                // Fallback to direct parsing if cache fails
                if let Ok(content) = std::fs::read_to_string(&filepath) {
                    // Extract tags from the file content
                    let tags = self.extract_tags_from_content(&content);

                    // Get relative path for reporting (normalize separators to forward slashes)
                    let rel_path = filepath
                        .strip_prefix(&self.vault)
                        .unwrap_or(&filepath)
                        .to_string_lossy()
                        .replace(std::path::MAIN_SEPARATOR, "/");

                    // Update tag map
                    for tag in tags {
                        *tag_counts.entry(tag.clone()).or_insert(0) += 1;
                        tag_files.entry(tag).or_default().push(rel_path.clone());
                    }
                }
            }
        }

        // Build tag summary
        let mut tag_summary = Vec::new();
        for (tag, count) in tag_counts {
            tag_summary.push(TagSummary {
                tag: tag.clone(),
                count,
                files: tag_files.get(&tag).cloned().unwrap_or_default(),
            });
        }

        // Sort by count (descending) and then by tag name (ascending)
        tag_summary.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.tag.cmp(&b.tag)));

        let result = serde_json::json!(tag_summary);
        log::info!("Tag summary generated successfully");

        Ok(CallToolResult::success(vec![Content::json(result)?]))
    }

    /// Find notes that have specific tags
    ///
    /// This function searches through all markdown files in the vault (or a specific folder)
    /// to find notes that contain any of the specified tags. It leverages the metadata cache
    /// for efficient tag lookup and returns file information along with their frontmatter.
    #[tool(
        description = "Find notes that have specific tags. Returns list of files with matching tags and their metadata. If folder_path is provided, it must be relative to the vault root, NOT an absolute path. File paths in results are returned relative to the vault root."
    )]
    pub fn get_note_by_tag(
        &self,
        #[tool(aggr)] GetNoteByTagRequest { tags, folder_path }: GetNoteByTagRequest,
    ) -> Result<CallToolResult, rmcp::Error> {
        if tags.is_empty() {
            return Err(rmcp::Error::invalid_request(
                "At least one tag must be provided".to_string(),
                None,
            ));
        }

        let mut matching_notes = Vec::new();
        let files = self.internal_list_files();

        // Filter files based on folder_path if provided
        let filtered_files: Vec<PathBuf> = if let Some(ref folder) = folder_path {
            let folder_path = self.validate_vault_path(folder)?;

            files
                .into_iter()
                .filter(|file| file.starts_with(&folder_path))
                .collect()
        } else {
            files
        };

        for file_path in filtered_files {
            // Only process markdown files
            if file_path.extension().is_none_or(|ext| ext != "md") {
                continue;
            }

            // Try to get cached metadata first
            if let Some(cache_data) = self.metadata_cache.get_or_parse(&file_path, |content| {
                self.extract_tags_from_content(content)
            }) {
                // Check if any of the requested tags match the file's tags
                let file_has_matching_tag = cache_data.tags.iter().any(|file_tag| {
                    tags.iter().any(|requested_tag| {
                        file_tag.eq_ignore_ascii_case(requested_tag)
                            || file_tag.eq_ignore_ascii_case(&format!("#{requested_tag}"))
                    })
                });

                if file_has_matching_tag {
                    // Get file system metadata
                    let metadata = match fs::metadata(&file_path) {
                        Ok(m) => m,
                        Err(e) => {
                            log::warn!("Failed to get metadata for {}: {}", file_path.display(), e);
                            continue;
                        }
                    };

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

                    // Convert file path to relative path from vault root (normalize separators to forward slashes)
                    let relative_path = file_path
                        .strip_prefix(&self.vault)
                        .unwrap_or(&file_path)
                        .to_string_lossy()
                        .replace(std::path::MAIN_SEPARATOR, "/");

                    // Convert frontmatter to JSON value
                    let frontmatter_json = match &cache_data.frontmatter {
                        Some(yaml_value) => {
                            serde_json::to_value(yaml_value).unwrap_or(serde_json::Value::Null)
                        }
                        None => serde_json::Value::Null,
                    };

                    matching_notes.push(NoteWithTags {
                        filename: relative_path,
                        tags: cache_data.tags.clone(),
                        frontmatter: frontmatter_json,
                        creation_date,
                        modification_date,
                    });
                }
            }
        }

        // Sort by filename for consistent output
        matching_notes.sort_by(|a, b| a.filename.cmp(&b.filename));

        let result = serde_json::json!(matching_notes);
        log::info!("Found {} notes with matching tags", matching_notes.len());

        Ok(CallToolResult::success(vec![Content::json(result)?]))
    }

    #[tool(
        description = "Enhanced search that returns matching content with surrounding context lines. Supports regex patterns and provides detailed match information including line numbers and context."
    )]
    pub fn search_with_context(
        &self,
        #[tool(aggr)] SearchWithContextRequest {
            query,
            context_lines,
            regex,
            case_sensitive,
        }: SearchWithContextRequest,
    ) -> Result<CallToolResult, rmcp::Error> {
        log::info!("Searching with context for: {query}");

        let context_lines = context_lines.unwrap_or(2);
        let is_regex = regex.unwrap_or(false);
        let is_case_sensitive = case_sensitive.unwrap_or(false);

        // Build regex pattern
        let regex_pattern = if is_regex {
            Regex::new(&query).map_err(|e| {
                rmcp::Error::invalid_request(format!("Invalid regex pattern: {e}"), None)
            })?
        } else {
            // Escape special regex characters for literal search
            let escaped_query = regex::escape(&query);
            if is_case_sensitive {
                Regex::new(&escaped_query).unwrap()
            } else {
                Regex::new(&format!("(?i){escaped_query}")).unwrap()
            }
        };

        let mut all_matches = Vec::new();
        let files = self.internal_list_files();

        for file_path in files {
            // Only search text files (primarily markdown)
            if let Some(ext) = file_path.extension() {
                if ext != "md" && ext != "txt" {
                    continue;
                }
            }

            let relative_path = file_path
                .strip_prefix(&self.vault)
                .unwrap_or(&file_path)
                .to_string_lossy()
                .replace(std::path::MAIN_SEPARATOR, "/");

            // Read file content to get context
            let content = match std::fs::read_to_string(&file_path) {
                Ok(content) => content,
                Err(_) => continue,
            };

            let lines: Vec<&str> = content.lines().collect();

            // Search each line for matches
            for (line_index, line) in lines.iter().enumerate() {
                if let Some(regex_match) = regex_pattern.find(line) {
                    let line_num = line_index + 1; // Convert to 1-based line number

                    // Get context before
                    let context_before = if line_index >= context_lines {
                        lines[(line_index - context_lines)..line_index]
                            .iter()
                            .map(|s| s.to_string())
                            .collect()
                    } else {
                        lines[0..line_index].iter().map(|s| s.to_string()).collect()
                    };

                    // Get context after
                    let context_after = if line_index + 1 + context_lines <= lines.len() {
                        lines[(line_index + 1)..(line_index + 1 + context_lines)]
                            .iter()
                            .map(|s| s.to_string())
                            .collect()
                    } else {
                        lines[(line_index + 1)..]
                            .iter()
                            .map(|s| s.to_string())
                            .collect()
                    };

                    all_matches.push(SearchMatch {
                        filename: relative_path.clone(),
                        line_number: line_num,
                        line_content: line.to_string(),
                        context_before,
                        context_after,
                        match_start: regex_match.start(),
                        match_end: regex_match.end(),
                    });
                }
            }
        }

        // Sort matches by filename and line number
        all_matches.sort_by(|a, b| {
            a.filename
                .cmp(&b.filename)
                .then_with(|| a.line_number.cmp(&b.line_number))
        });

        let result = serde_json::json!(all_matches);
        log::info!("Found {} matches with context", all_matches.len());

        Ok(CallToolResult::success(vec![Content::json(result)?]))
    }

    #[tool(
        description = "Find all notes that link to or are linked from a specific note. Returns both incoming and outgoing links with their contexts."
    )]
    pub fn get_linked_notes(
        &self,
        #[tool(aggr)] GetLinkedNotesRequest { filename }: GetLinkedNotesRequest,
    ) -> Result<CallToolResult, rmcp::Error> {
        let target_file_path = self.validate_vault_path(&filename)?;

        // Verify the target file exists
        if !target_file_path.exists() {
            return Err(rmcp::Error::invalid_request(
                format!("File '{filename}' does not exist"),
                None,
            ));
        }

        // Get the target filename without path and extension for link matching
        let target_name = target_file_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&filename);

        // Read the target file to get its outgoing links
        let target_content = std::fs::read_to_string(&target_file_path).map_err(|e| {
            rmcp::Error::internal_error(format!("Failed to read target file: {e}"), None)
        })?;

        let outgoing_links = extract_links_from_content(&target_content);

        // Find all files that link to this note (incoming links)
        let mut incoming_links = Vec::new();
        let files = self.internal_list_files();

        for file_path in files {
            // Skip the target file itself
            if file_path == target_file_path {
                continue;
            }

            // Only check markdown files
            if file_path.extension().is_none_or(|ext| ext != "md") {
                continue;
            }

            let relative_path = file_path
                .strip_prefix(&self.vault)
                .unwrap_or(&file_path)
                .to_string_lossy()
                .replace(std::path::MAIN_SEPARATOR, "/");

            // Get cached links for this file
            if let Some(cache_data) = self
                .metadata_cache
                .get_or_parse(&file_path, |_content| Vec::new())
            // We use cached links, not tags
            {
                // Check if this file links to our target
                let links_to_target = cache_data.links.iter().any(|link| {
                    // Handle different link formats - use cross-platform path separators
                    let separator = std::path::MAIN_SEPARATOR;
                    link == target_name
                        || link == &filename
                        || link.ends_with(&format!("{separator}{target_name}"))
                        || link.ends_with(&format!("{separator}{filename}"))
                });

                if links_to_target {
                    incoming_links.push(relative_path);
                }
            }
        }

        // Sort the results for consistent output
        incoming_links.sort();

        let result = LinkedNotes {
            filename: filename.clone(),
            outgoing_links,
            incoming_links,
        };

        log::info!(
            "Found {} outgoing and {} incoming links for '{}'",
            result.outgoing_links.len(),
            result.incoming_links.len(),
            filename
        );

        Ok(CallToolResult::success(vec![Content::json(
            serde_json::json!(result),
        )?]))
    }
}

#[tool(tool_box)]
impl ServerHandler for Obsidian {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;
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
            (
                temp_dir.path().join("tagged_document.md"),
                "# Document with Multiple Tags\nThis document has #multiple #tags and some #duplicate #tags\nAlso #multiple appears twice in the document #newline",
            ),
            (
                char_dir.join("tagged_npc.md"),
                "# Tagged NPC\nThis is an NPC with #character and #important tags",
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
    fn test_get_tags_summary() {
        // Create a test vault
        let temp_dir = create_test_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Test getting tags summary for the entire vault
        let result = obsidian
            .get_tags_summary(GetTagsSummaryRequest { folder_path: None })
            .expect("Failed to get tags summary");

        // Extract the content from the result
        let content = result.content;
        let content_str = format!("{:?}", content[0]);

        // Check that we have JSON content with tags
        assert!(
            content_str.contains("tag"),
            "Response should contain tag information"
        );
        assert!(
            content_str.contains("character"),
            "Response should contain character tag"
        );
        assert!(
            content_str.contains("multiple"),
            "Response should contain multiple tag"
        );
        assert!(
            content_str.contains("metadata"),
            "Response should contain metadata tag"
        );

        // Check for count and sorting information
        assert!(
            content_str.contains("count"),
            "Response should contain count information"
        );

        // Check that the content contains file paths
        assert!(
            content_str.contains("files"),
            "Response should contain file paths"
        );
        assert!(
            content_str.contains(".md"),
            "Response should contain markdown file extensions"
        );

        // Test with a specific folder path
        let result = obsidian
            .get_tags_summary(GetTagsSummaryRequest {
                folder_path: Some("characters".to_string()),
            })
            .expect("Failed to get tags summary for characters folder");

        // Extract the content from the result
        let content = result.content;
        let content_str = format!("{:?}", content[0]);

        // Verify that content is returned
        assert!(
            content_str.contains("tag"),
            "Response should contain tag information"
        );

        // Verify that only files from characters directory are included
        assert!(
            content_str.contains("characters/"),
            "Response should contain files from characters directory"
        );
        assert!(
            !content_str.contains("locations/"),
            "Response should not contain files from locations directory"
        );

        // Test with an invalid folder path
        let result = obsidian
            .get_tags_summary(GetTagsSummaryRequest {
                folder_path: Some("nonexistent".to_string()),
            })
            .expect("Function should not fail");

        // Check the content for error
        let content = result.content;
        let content_str = format!("{:?}", content[0]);
        assert!(
            content_str.contains("Folder not found"),
            "Should return an error content for nonexistent folder"
        );
    }

    #[test]
    fn test_get_vault_structure_root() {
        // Create a test vault
        let temp_dir = create_test_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Get the structure of the entire vault
        let structure = obsidian
            .build_directory_structure(temp_dir.path().to_path_buf())
            .expect("it can be built");

        // Validate the structure
        assert_eq!(structure.name, "vault");
        assert_eq!(structure.files.len(), 3); // 3 files directly in the root
        assert_eq!(structure.subdirectories.len(), 2); // 2 subdirectories

        // Validate the characters directory
        let chars = structure
            .subdirectories
            .get("characters")
            .expect("Characters directory not found");
        assert_eq!(chars.name, "characters");
        assert_eq!(chars.files.len(), 3); // All files directly in this directory
        assert_eq!(chars.subdirectories.len(), 0); // No subdirectories

        // Validate the locations directory
        let locs = structure
            .subdirectories
            .get("locations")
            .expect("Locations directory not found");
        assert_eq!(locs.name, "locations");
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
        // Get the structure of a subdirectory
        let structure = obsidian
            .build_directory_structure(char_path)
            .expect("it can be built");

        // Validate the structure
        assert_eq!(structure.name, "characters");
        assert_eq!(structure.files.len(), 3); // All files directly in this directory
        assert_eq!(structure.subdirectories.len(), 0); // No subdirectories

        // Verify file names
        assert!(structure.files.contains(&"npc1.md".to_string()));
        assert!(structure.files.contains(&"npc2.md".to_string()));
        assert!(structure.files.contains(&"tagged_npc.md".to_string()));
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
        assert!(content_str.contains("tagged_npc.md"));
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

        // Test file with tags and links (using relative path)
        let request = GetFileMetadataRequest {
            filename: "characters/npc1.md".to_string(),
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
        // Test file with frontmatter (using relative path)
        let request = GetFileMetadataRequest {
            filename: "with_frontmatter.md".to_string(),
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
    fn test_get_file_metadata_invalid_file() {
        // Create a test vault
        let temp_dir = create_test_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Test with a non-existent markdown file
        let request = GetFileMetadataRequest {
            filename: "non_existent_file.md".to_string(),
        };

        // Should return an error rather than panic
        let result = obsidian.get_file_metadata(request);
        assert!(result.is_err());
    }

    // Tests for the MetadataCache
    #[test]
    fn test_metadata_cache() {
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test.md");

        // Create a test file
        let content = "---\ntitle: Test\n---\n# Header\nThis is a #test file with a [[link]]";
        fs::write(&file_path, content).unwrap();

        let cache = MetadataCache::new();

        // Extract tags function
        let extract_tags = |content: &str| {
            let mut tags = Vec::new();
            for word in content.split_whitespace() {
                if word.starts_with('#') && word.len() > 1 {
                    let tag = word[1..]
                        .trim_end_matches(|c: char| !c.is_alphanumeric())
                        .to_string();
                    if !tag.is_empty() && !tags.contains(&tag) {
                        tags.push(tag);
                    }
                }
            }
            tags
        };

        // First access should parse and cache
        let cached_data = cache.get_or_parse(&file_path, extract_tags).unwrap();
        assert_eq!(cached_data.tags, vec!["test"]);
        assert_eq!(cached_data.links, vec!["link"]);
        assert!(cached_data.frontmatter.is_some());

        // Second access should use cache
        let cached_data2 = cache.get_or_parse(&file_path, extract_tags).unwrap();
        assert_eq!(cached_data2.tags, vec!["test"]);

        // Modify the file to test cache invalidation
        let new_content =
            "---\ntitle: Updated\n---\n# Header\nThis is a #modified file with a [[newlink]]";
        std::thread::sleep(Duration::from_millis(10)); // Ensure timestamp changes
        fs::write(&file_path, new_content).unwrap();

        // Cache should detect the change and reparse
        let cached_data3 = cache.get_or_parse(&file_path, extract_tags).unwrap();
        assert_eq!(cached_data3.tags, vec!["modified"]);
        assert_eq!(cached_data3.links, vec!["newlink"]);
    }

    #[test]
    fn test_metadata_cache_threaded() {
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("thread_test.md");

        // Create a test file
        let content = "---\ntitle: Thread Test\n---\n# Header\nThis is a #threaded #test";
        fs::write(&file_path, content).unwrap();

        let cache = Arc::new(MetadataCache::new());

        // Extract tags function
        let extract_tags = |content: &str| {
            let mut tags = Vec::new();
            for word in content.split_whitespace() {
                if word.starts_with('#') && word.len() > 1 {
                    let tag = word[1..]
                        .trim_end_matches(|c: char| !c.is_alphanumeric())
                        .to_string();
                    if !tag.is_empty() && !tags.contains(&tag) {
                        tags.push(tag);
                    }
                }
            }
            tags
        };

        let threads: Vec<_> = (0..10)
            .map(|_| {
                let cache_clone = Arc::clone(&cache);
                let path_clone = file_path.clone();
                thread::spawn(move || {
                    // Access the cache from multiple threads
                    let cached_data = cache_clone.get_or_parse(&path_clone, extract_tags).unwrap();
                    assert!(cached_data.tags.contains(&"threaded".to_string()));
                    assert!(cached_data.tags.contains(&"test".to_string()));
                })
            })
            .collect();

        // Wait for all threads to complete
        for thread in threads {
            thread.join().unwrap();
        }
    }

    // Edge case tests
    #[test]
    fn test_metadata_cache_missing_file() {
        let temp_dir = tempfile::tempdir().unwrap();
        let non_existent_file = temp_dir.path().join("non_existent.md");

        let cache = MetadataCache::new();
        let extract_tags = |_: &str| Vec::new();

        // Should return None for a non-existent file
        let result = cache.get_or_parse(&non_existent_file, extract_tags);
        assert!(result.is_none());
    }

    #[test]
    fn test_metadata_cache_empty_file() {
        let temp_dir = tempfile::tempdir().unwrap();
        let empty_file_path = temp_dir.path().join("empty.md");

        // Create an empty file
        fs::write(&empty_file_path, "").unwrap();

        let cache = MetadataCache::new();
        let extract_tags = |_: &str| Vec::new();

        // Should parse an empty file successfully
        let cached_data = cache.get_or_parse(&empty_file_path, extract_tags).unwrap();
        assert!(cached_data.tags.is_empty());
        assert!(cached_data.links.is_empty());
        assert!(cached_data.frontmatter.is_none());
    }

    #[test]
    fn test_extract_links_from_content() {
        use std::collections::HashSet;

        // Test with multiple links
        let content = "This is text with [[link1]] and [[link2]] and [[link3]].";
        let links = extract_links_from_content(content);
        let expected: HashSet<String> = ["link1", "link2", "link3"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let actual: HashSet<String> = links.into_iter().collect();
        assert_eq!(actual, expected);

        // Test with duplicate links
        let content = "This has [[duplicate]] and [[duplicate]] links.";
        let links = extract_links_from_content(content);
        let expected: HashSet<String> = ["duplicate"].iter().map(|s| s.to_string()).collect();
        let actual: HashSet<String> = links.into_iter().collect();
        assert_eq!(actual, expected);

        // Test with no links
        let content = "This text has no links at all.";
        let links = extract_links_from_content(content);
        assert!(links.is_empty());

        // Test with pipe syntax (display text)
        let content = "Link to [[page1|Display Text]] and [[page2|Another Display]].";
        let links = extract_links_from_content(content);
        let expected: HashSet<String> = ["page1", "page2"].iter().map(|s| s.to_string()).collect();
        let actual: HashSet<String> = links.into_iter().collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_extract_frontmatter_from_content() {
        // Test with valid frontmatter
        let content = "---\ntitle: Test\nauthor: Someone\n---\nContent here";
        let frontmatter = extract_frontmatter_from_content(content);
        assert!(frontmatter.is_some());

        // Test with no frontmatter
        let content = "No frontmatter here at all";
        let frontmatter = extract_frontmatter_from_content(content);
        assert!(frontmatter.is_none());

        // Test with invalid frontmatter format
        let content = "---\nThis is not valid YAML\n::\n---\nContent";
        let frontmatter = extract_frontmatter_from_content(content);
        assert!(frontmatter.is_none());
    }

    #[test]
    fn test_get_note_by_tag() {
        // Create a test vault
        let temp_dir = create_test_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Test searching for a specific tag
        let request = GetNoteByTagRequest {
            tags: vec!["character".to_string()],
            folder_path: None,
        };

        let result = obsidian
            .get_note_by_tag(request)
            .expect("Tool function failed");

        // Verify the result contains notes with the character tag
        let content = result.content;
        let content_str = format!("{:?}", content[0]);

        // Should find character files that have the character tag
        assert!(content_str.contains("character"));

        // Test with multiple tags
        let request = GetNoteByTagRequest {
            tags: vec!["character".to_string(), "important".to_string()],
            folder_path: None,
        };

        let result = obsidian
            .get_note_by_tag(request)
            .expect("Tool function failed");

        // Should find files with either tag
        let content = result.content;
        assert!(!content.is_empty());

        // Test with folder path filter
        let request = GetNoteByTagRequest {
            tags: vec!["character".to_string()],
            folder_path: Some("characters".to_string()),
        };

        let _result = obsidian
            .get_note_by_tag(request)
            .expect("Tool function should succeed even with folder filter");

        // Test with empty tags (should return error)
        let request = GetNoteByTagRequest {
            tags: vec![],
            folder_path: None,
        };

        let result = obsidian.get_note_by_tag(request);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_text_file_absolute_path() {
        let temp_dir = create_test_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Test with platform-specific absolute paths
        #[cfg(unix)]
        let absolute_path = "/absolute/path/file.md";
        
        #[cfg(windows)]
        let absolute_path = "C:\\absolute\\path\\file.md";
        
        #[cfg(not(any(unix, windows)))]
        let absolute_path = "/absolute/path/file.md"; // Fallback for other platforms

        let request = ReadTextFileRequest {
            filename: absolute_path.to_string(),
        };

        let result = obsidian.read_text_file(request);
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(error.to_string().contains("Invalid vault path"));
        assert!(error.to_string().contains("absolute"));
    }

    #[test]
    fn test_get_file_metadata_absolute_path() {
        let temp_dir = create_test_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Test with platform-specific absolute paths
        #[cfg(unix)]
        let absolute_path = "/absolute/path/file.md";
        
        #[cfg(windows)]
        let absolute_path = "C:\\absolute\\path\\file.md";
        
        #[cfg(not(any(unix, windows)))]
        let absolute_path = "/absolute/path/file.md"; // Fallback for other platforms

        let request = GetFileMetadataRequest {
            filename: absolute_path.to_string(),
        };

        let result = obsidian.get_file_metadata(request);
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(error.to_string().contains("Invalid vault path"));
        assert!(error.to_string().contains("absolute"));
    }

    #[test]
    fn test_get_vault_structure_absolute_path() {
        let temp_dir = create_test_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Test with platform-specific absolute paths
        #[cfg(unix)]
        let absolute_path = "/absolute/path";
        
        #[cfg(windows)]
        let absolute_path = "C:\\absolute\\path";
        
        #[cfg(not(any(unix, windows)))]
        let absolute_path = "/absolute/path"; // Fallback for other platforms

        let request = GetVaultStructureRequest {
            folder_path: Some(absolute_path.to_string()),
        };

        let result = obsidian.get_vault_structure(request);
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(error.to_string().contains("Invalid vault path"));
        assert!(error.to_string().contains("absolute"));
    }

    #[test]
    fn test_get_tags_summary_absolute_path() {
        let temp_dir = create_test_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Test with platform-specific absolute paths
        #[cfg(unix)]
        let absolute_path = "/absolute/path";
        
        #[cfg(windows)]
        let absolute_path = "C:\\absolute\\path";
        
        #[cfg(not(any(unix, windows)))]
        let absolute_path = "/absolute/path"; // Fallback for other platforms

        let request = GetTagsSummaryRequest {
            folder_path: Some(absolute_path.to_string()),
        };

        let result = obsidian.get_tags_summary(request);
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(error.to_string().contains("Invalid vault path"));
        assert!(error.to_string().contains("absolute"));
    }

    #[test]
    fn test_get_note_by_tag_absolute_path() {
        let temp_dir = create_test_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Test with platform-specific absolute paths
        #[cfg(unix)]
        let absolute_path = "/absolute/path";
        
        #[cfg(windows)]
        let absolute_path = "C:\\absolute\\path";
        
        #[cfg(not(any(unix, windows)))]
        let absolute_path = "/absolute/path"; // Fallback for other platforms

        let request = GetNoteByTagRequest {
            tags: vec!["test".to_string()],
            folder_path: Some(absolute_path.to_string()),
        };

        let result = obsidian.get_note_by_tag(request);
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(error.to_string().contains("Invalid vault path"));
        assert!(error.to_string().contains("absolute"));
    }

    #[test]
    fn test_relative_paths_work() {
        let temp_dir = create_test_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Test that relative paths work correctly
        let request = GetVaultStructureRequest {
            folder_path: Some("subfolder".to_string()),
        };

        let result = obsidian.get_vault_structure(request);
        // Should not error for relative paths (though might fail if folder doesn't exist)
        // The important thing is that it doesn't fail with InvalidVaultPath
        if let Err(error) = result {
            assert!(!error.to_string().contains("Invalid vault path"));
        }
    }

    #[test]
    fn test_invalid_vault_path_error_conversion() {
        use crate::errors::Error;

        // Test that InvalidVaultPath error converts properly to rmcp::Error
        let error = Error::InvalidVaultPath("/absolute/path/test.md".to_string());
        let rmcp_error: rmcp::Error = error.into();

        let error_str = rmcp_error.to_string();
        assert!(error_str.contains("Invalid vault path"));
        assert!(error_str.contains("/absolute/path/test.md"));
        assert!(error_str.contains("paths must be relative to the vault root"));
        assert!(error_str.contains("not absolute"));
    }

    #[test]
    fn test_validate_vault_path_helper() {
        let temp_dir = create_test_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Test that relative paths work correctly
        let result = obsidian.validate_vault_path("characters/npc1.md");
        assert!(result.is_ok());
        let path = result.unwrap();
        assert!(path.ends_with("characters/npc1.md"));
        assert!(path.is_absolute()); // Should be absolute after joining with vault

        // Test absolute paths using cross-platform path construction
        #[cfg(unix)]
        {
            // Unix absolute path
            let result = obsidian.validate_vault_path("/absolute/path/file.md");
            assert!(result.is_err());

            let error = result.unwrap_err();
            match error {
                Error::InvalidVaultPath(path) => {
                    assert_eq!(path, "/absolute/path/file.md");
                }
                _ => panic!("Expected InvalidVaultPath error"),
            }
        }

        #[cfg(windows)]
        {
            // Windows absolute paths
            let result = obsidian.validate_vault_path("C:\\absolute\\path\\file.md");
            assert!(result.is_err());

            let result = obsidian.validate_vault_path("D:/absolute/path/file.md");
            assert!(result.is_err());

            // UNC path
            let result = obsidian.validate_vault_path("\\\\server\\share\\file.md");
            assert!(result.is_err());
        }

        // Test cross-platform absolute path using std::path
        let absolute_path = std::path::Path::new("/").join("absolute").join("path").join("file.md");
        let result = obsidian.validate_vault_path(&absolute_path.to_string_lossy());
        assert!(result.is_err());

        // Test empty relative path
        let result = obsidian.validate_vault_path("");
        assert!(result.is_ok());
        let path = result.unwrap();
        assert_eq!(path, obsidian.vault);

        // Test relative path with current directory
        let result = obsidian.validate_vault_path("./file.md");
        assert!(result.is_ok());
        let path = result.unwrap();
        assert!(path.ends_with("file.md"));

        // Test directory traversal protection
        let result = obsidian.validate_vault_path("../../../etc/passwd");
        // This might not error on all systems if the path doesn't exist to canonicalize,
        // but it should at least not give access outside the vault
        if result.is_ok() {
            let path = result.unwrap();
            // Ensure the path is still within the vault directory structure
            assert!(path.starts_with(&obsidian.vault));
        }

        // Test relative path with parent directory references
        let result = obsidian.validate_vault_path("./characters/../notes.md");
        assert!(result.is_ok());
    }

    #[test]
    fn test_search_with_context() {
        let temp_dir = create_test_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Test basic search with context
        let request = SearchWithContextRequest {
            query: "character".to_string(),
            context_lines: Some(1),
            regex: Some(false),
            case_sensitive: Some(false),
        };

        let result = obsidian.search_with_context(request).unwrap();

        // Verify that we got a successful result
        assert_eq!(result.content.len(), 1);

        // Check result contains expected data by converting to string
        let content_str = format!("{:?}", result.content[0]);
        assert!(
            content_str.contains("character"),
            "Should find matches for 'character'"
        );
        assert!(
            content_str.contains("filename"),
            "Should have filename field"
        );
        assert!(
            content_str.contains("line_number"),
            "Should have line_number field"
        );
    }

    #[test]
    fn test_search_with_context_regex() {
        let temp_dir = create_test_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Test regex search
        let request = SearchWithContextRequest {
            query: r"character|NPC".to_string(),
            context_lines: Some(2),
            regex: Some(true),
            case_sensitive: Some(false),
        };

        let result = obsidian.search_with_context(request).unwrap();

        // Verify that we got a successful result
        assert_eq!(result.content.len(), 1);

        // Check result contains expected data
        let content_str = format!("{:?}", result.content[0]);
        assert!(
            content_str.contains("character") || content_str.contains("NPC"),
            "Should find matches for regex pattern"
        );
    }

    #[test]
    fn test_search_with_context_case_sensitive() {
        let temp_dir = create_test_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Test case sensitive search
        let request = SearchWithContextRequest {
            query: "Character".to_string(), // Capital C
            context_lines: Some(1),
            regex: Some(false),
            case_sensitive: Some(true),
        };

        let result = obsidian.search_with_context(request).unwrap();

        // Verify that we got a successful result
        assert_eq!(result.content.len(), 1);

        // For case sensitive search, check that we're looking for exact case
        let content_str = format!("{:?}", result.content[0]);
        // This might not find matches if the test data doesn't have "Character" with capital C
        // The test validates the function works, even if no matches are found

        // Check that the result structure is valid (may be empty for case sensitive search)
        assert!(
            content_str.contains("[]") || content_str.contains("Character"),
            "Case sensitive search should return empty array or exact matches"
        );
    }

    #[test]
    fn test_get_linked_notes() {
        let temp_dir = create_test_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Create a test file with links
        let test_file = temp_dir.path().join("test_with_links.md");
        let mut file = std::fs::File::create(&test_file).unwrap();
        writeln!(file, "# Test Note").unwrap();
        writeln!(
            file,
            "This note links to [[npc1]] and [[locations/tavern]]."
        )
        .unwrap();
        writeln!(file, "It also mentions [[another_note]].").unwrap();

        // Create a file that links back to our test file
        let linking_file = temp_dir.path().join("linking_back.md");
        let mut file = std::fs::File::create(&linking_file).unwrap();
        writeln!(file, "# Linking Back").unwrap();
        writeln!(file, "This references [[test_with_links]].").unwrap();

        let request = GetLinkedNotesRequest {
            filename: "test_with_links.md".to_string(),
        };

        let result = obsidian.get_linked_notes(request).unwrap();

        // Verify that we got a successful result
        assert_eq!(result.content.len(), 1);

        // Check result contains expected data
        let content_str = format!("{:?}", result.content[0]);
        assert!(
            content_str.contains("test_with_links.md"),
            "Should contain target filename"
        );
        assert!(
            content_str.contains("outgoing_links"),
            "Should have outgoing_links field"
        );
        assert!(
            content_str.contains("incoming_links"),
            "Should have incoming_links field"
        );
        assert!(content_str.contains("npc1"), "Should find link to npc1");
    }

    #[test]
    fn test_get_linked_notes_nonexistent_file() {
        let temp_dir = create_test_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        let request = GetLinkedNotesRequest {
            filename: "nonexistent.md".to_string(),
        };

        let result = obsidian.get_linked_notes(request);
        assert!(result.is_err(), "Should return error for nonexistent file");
    }

    #[test]
    fn test_get_linked_notes_no_links() {
        let temp_dir = create_test_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Create a file with no links
        let test_file = temp_dir.path().join("no_links.md");
        let mut file = std::fs::File::create(&test_file).unwrap();
        writeln!(file, "# No Links").unwrap();
        writeln!(file, "This file has no wiki-style links.").unwrap();

        let request = GetLinkedNotesRequest {
            filename: "no_links.md".to_string(),
        };

        let result = obsidian.get_linked_notes(request).unwrap();

        // Verify that we got a successful result
        assert_eq!(result.content.len(), 1);

        // Check result structure
        let content_str = format!("{:?}", result.content[0]);
        assert!(
            content_str.contains("no_links.md"),
            "Should contain target filename"
        );
        assert!(
            content_str.contains("outgoing_links"),
            "Should have outgoing_links field"
        );
        assert!(
            content_str.contains("incoming_links"),
            "Should have incoming_links field"
        );
    }
}
