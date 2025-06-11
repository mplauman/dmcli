# Enhanced MCP Server Functions for DM Notes Collection

This document outlines a comprehensive set of functions for an MCP server that would allow an LLM to quickly and efficiently understand a collection of Dungeon Master's notes.

## Implementation Progress
- ✅ Complete: 6/18 functions (33%)
- ❌ Pending: 12/18 functions (67%)

## Current Implementation Status

The existing implementation includes:
1. `list_files` - Lists all files recursively ✅
2. `grep` - Simple text pattern search ✅
3. `read_text_file` - Reads a specific file's contents ✅
4. `get_vault_structure` - Returns hierarchical folder structure ✅
5. `get_file_metadata` - Extracts metadata from markdown files ✅
6. `get_tags_summary` - Returns tags with frequency ✅

These provide basic functionality, but more advanced features are still needed for optimal understanding of the semantic structure of DM notes.

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

#### 2.1. `get_note_by_tag` ❌
- Find notes that have specific tags
- Parameters: Tag name(s)
- Returns: List of files with matching tags and their frontmatter

#### 2.2. `search_with_context` ❌
- Enhanced version of grep that returns matching content with surrounding context
- Parameters: Search query, context lines (before/after)
- Returns: Snippets of text with matches highlighted, including file metadata

#### 2.3. `get_linked_notes` ❌
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

## Example Implementation of Key Function

Here's how one of these functions might be implemented in the existing codebase:

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