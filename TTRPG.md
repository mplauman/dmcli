# Enhanced MCP Server Functions for DM Notes Collection

This document outlines a comprehensive set of functions for an MCP server that would allow an LLM to quickly and efficiently understand a collection of Dungeon Master's notes.

## Implementation Progress
- ✅ Complete: 8/17 functions (47%)
- ❌ Pending: 9/17 functions (53%)

Note: The old `grep` function has been replaced by the more powerful `search_with_context` function.

## Current Implementation Status

The existing implementation includes:
1. `list_files` - Lists all files recursively ✅
2. `read_text_file` - Reads a specific file's contents ✅
3. `get_vault_structure` - Returns hierarchical folder structure ✅
4. `get_file_metadata` - Extracts metadata from markdown files ✅
5. `get_tags_summary` - Returns tags with frequency ✅
6. `get_note_by_tag` - Find notes with specific tags ✅
7. `search_with_context` - Enhanced search with surrounding context lines ✅
8. `get_linked_notes` - Find incoming and outgoing note links ✅

These provide comprehensive functionality for understanding and navigating DM note collections, with advanced search and relationship mapping capabilities.

## Proposed Enhanced Functions

### 1. Metadata and Structure Functions

#### 1.1. `get_vault_structure` ✅
- Returns a hierarchical representation of the vault's folder structure
- Helps the LLM understand how the notes are organized
- Parameters: Optional folder path to get structure of a subsection
- Returns: JSON representation of the directory structure with file counts

#### 1.2. `get_file_metadata` ✅
- Extracts front matter and metadata from markdown files
- Parameters: Filename
- Returns: Creation date, modification date, tags, links, frontmatter properties

#### 1.3. `get_tags_summary` ✅
- Returns all tags used across the vault with frequency
- Helps the LLM understand the tagging system and common themes
- Parameters: Optional folder path to limit scope
- Returns: List of tags with counts

### 2. Enhanced Content Retrieval

#### 2.1. `get_note_by_tag` ✅
- Find notes that have specific tags
- Parameters: Tag name(s)
- Returns: List of files with matching tags and their frontmatter

#### 2.2. `search_with_context` ✅
- Enhanced version of grep that returns matching content with surrounding context
- Parameters: Search query, context lines (before/after), regex support, case sensitivity
- Returns: Snippets of text with matches highlighted, including file metadata and line numbers

#### 2.3. `get_linked_notes` ✅
- Find all notes that link to or are linked from a specific note
- Parameters: Filename
- Returns: Incoming and outgoing links with context

### 3. TTRPG-Specific Functions

#### 3.1. `get_campaign_overview` ❌
- Extract campaign summary information
- Parameters: Campaign name (if multiple campaigns exist)
- Returns: Main campaign notes, session list, key characters and locations

#### 3.2. `get_npcs` ❌
- Retrieve all NPCs mentioned in notes
- Parameters: Optional location or faction filter
- Returns: Structured information about NPCs, their descriptions, and relationships

#### 3.3. `get_locations` ❌
- Retrieve all locations mentioned in notes
- Parameters: Optional region or type filter
- Returns: Structured information about locations, their descriptions, and connections

#### 3.4. `get_session_notes` ❌
- Retrieve notes for specific session(s)
- Parameters: Session number or date range
- Returns: Session notes with before/after context

### 4. Knowledge Graph Functions

#### 4.1. `get_entity_relationships` ❌
- Extract relationship graphs between entities (NPCs, locations, items)
- Parameters: Optional entity name or type
- Returns: Graph of relationships between entities

#### 4.2. `get_timeline` ❌
- Extract chronological events from notes
- Parameters: Optional date range or entity filter
- Returns: Timeline of events mentioned in notes

### 5. Semantic Understanding Functions

#### 5.1. `summarize_notes` ❌
- Generate summaries for notes or sections
- Parameters: Path or filter criteria, summary length
- Returns: Concise summaries of the requested content

#### 5.2. `extract_entities` ❌
- Identify and extract entities mentioned in notes (characters, locations, items)
- Parameters: Filename or search pattern
- Returns: Structured list of entities with their contexts

#### 5.3. `classify_notes` ❌
- Categorize notes by their primary purpose (session prep, location description, NPC details, etc.)
- Parameters: Optional folder path
- Returns: Categorized list of notes

## Implementation Considerations

1. **Incremental Complexity**: These functions could be implemented progressively, starting with extensions to the current system. (Progress: 3/15 functions implemented)

2. **Markdown Parsing**: A robust markdown parser would be needed to extract structured information.

3. **Caching**: For performance, implement caching of file contents and metadata.

4. **Frontmatter Handling**: Add special handling for YAML frontmatter which is common in Obsidian notes.

5. **Lightweight Indexing**: Create a lightweight index of the vault to speed up queries.

## Implementation Details for Completed Functions

### `search_with_context` Function

This function provides enhanced search capabilities with contextual information around matches:

#### Features:
- **Regex Support**: Can perform both literal string searches and regex pattern matching
- **Case Sensitivity**: Configurable case-sensitive or case-insensitive search
- **Context Lines**: Returns specified number of lines before and after each match
- **Match Positioning**: Provides exact character positions of matches within lines
- **File Filtering**: Only searches text files (primarily .md and .txt)

#### Request Structure:
```rust
pub struct SearchWithContextRequest {
    pub query: String,                    // Search pattern
    pub context_lines: Option<usize>,     // Lines of context (default: 2)
    pub regex: Option<bool>,              // Enable regex (default: false)
    pub case_sensitive: Option<bool>,     // Case sensitivity (default: false)
}
```

#### Response Structure:
```rust
pub struct SearchMatch {
    pub filename: String,        // Relative path from vault root
    pub line_number: usize,      // 1-based line number
    pub line_content: String,    // The actual line with the match
    pub context_before: Vec<String>,  // Lines before the match
    pub context_after: Vec<String>,   // Lines after the match
    pub match_start: usize,      // Character position where match starts
    pub match_end: usize,        // Character position where match ends
}
```

#### Use Cases:
- Finding specific terms with surrounding context for better understanding
- Regex searches for complex patterns (e.g., finding all dates, names, etc.)
- Case-sensitive searches for proper nouns or specific formatting

### `get_linked_notes` Function

This function analyzes wiki-style links (`[[note_name]]`) to map relationships between notes:

#### Features:
- **Bidirectional Link Analysis**: Finds both outgoing and incoming links
- **Link Format Support**: Handles various wiki-link formats including paths
- **Metadata Caching**: Uses the existing cache system for performance
- **File Validation**: Ensures target file exists before processing

#### Request Structure:
```rust
pub struct GetLinkedNotesRequest {
    pub filename: String,  // Target file (relative to vault root)
}
```

#### Response Structure:
```rust
pub struct LinkedNotes {
    pub filename: String,           // The target filename
    pub outgoing_links: Vec<String>, // Notes this file links to
    pub incoming_links: Vec<String>, // Notes that link to this file
}
```

#### Link Detection Logic:
- Extracts `[[link_name]]` patterns from markdown content
- Handles nested paths like `[[locations/tavern]]`
- Matches links by filename with and without extensions
- Caches link data for improved performance on repeated queries

#### Use Cases:
- Understanding note relationships and dependencies
- Finding related content when preparing for sessions
- Identifying orphaned notes (no incoming links)
- Mapping knowledge graphs of campaign elements

## Example Implementation of Key Function

Here's how one of the remaining functions might be implemented in the existing codebase:

```rust
#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct GetNpcsRequest {
    #[schemars(description = "Optional location to filter NPCs by")]
    pub location: Option<String>,
    
    #[schemars(description = "Optional faction to filter NPCs by")]
    pub faction: Option<String>,
}

#[derive(serde::Serialize)]
pub struct NpcInfo {
    pub name: String,
    pub description: String,
    pub location: Option<String>,
    pub faction: Option<String>,
    pub relationships: Vec<String>,
    pub source_file: String,
}

#[tool(
    description = "retrieves information about NPCs (non-player characters) from the campaign notes"
)]
pub fn get_npcs(
    &self,
    #[tool(aggr)] GetNpcsRequest { location, faction }: GetNpcsRequest,
) -> Result<CallToolResult, rmcp::Error> {
    // Implementation would:
    // 1. Scan all markdown files for NPC information
    // 2. Parse frontmatter and content for structured NPC data
    // 3. Apply filters for location/faction if provided
    // 4. Format and return the data
    
    // This is simplified pseudocode:
    let files = self.internal_list_files();
    let mut npcs = Vec::<NpcInfo>::new();
    
    for path in files {
        if let Ok(content) = std::fs::read_to_string(&path) {
            // Parse the markdown, extract NPC information
            // Add to npcs collection if matching filters
        }
    }
    
    // Filter by location/faction if provided
    
    let result = serde_json::json!(npcs);
    Ok(CallToolResult::success(vec![Content::json(result)?]))
}
```

By implementing these functions, the LLM would be able to efficiently navigate and understand the DM's notes, making it a more effective assistant for campaign management, planning, and reference during gameplay.