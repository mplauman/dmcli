# dmcli - Dungeon Master Command Line Interface

A Terminal User Interface (TUI) application for Dungeon Masters to interact with AI assistants and manage campaign notes.

## Overview

dmcli provides a unified interface for interacting with Large Language Models (LLMs) to help with D&D campaign management. It includes integration with tools for searching through notes, managing campaign information, and generating content.

## Recent Changes

### Migration from Anthropic API to `llm` Crate

The application has been migrated from using the Anthropic API directly to using the unified `llm` crate, which provides:

- **Multi-backend support**: Can use OpenAI, Anthropic, Ollama, and other LLM providers
- **Unified interface**: Consistent API across different LLM backends
- **Tool calling**: Integrated function calling capabilities
- **Better error handling**: Improved error types and handling

### Key Changes

1. **Replaced `src/anthropic.rs`** with `src/llm_client.rs` 
   - Same public interface maintained for backward compatibility
   - Now supports multiple LLM backends via the `llm` crate

2. **Updated Configuration**
   - API keys and models can still be configured the same way
   - Configuration keys remain unchanged for compatibility

3. **Enhanced Tool Integration**
   - MCP (Model Context Protocol) tools are now converted to LLM function definitions
   - Supports parallel tool execution
   - Better tool result handling

## Configuration

The application expects configuration in `~/.config/dmcli.toml`:

```toml
[anthropic]
api_key = "your-api-key-here"
model = "claude-3-5-haiku-20241022"  # Optional, defaults to this
max_tokens = 8192                    # Optional, defaults to this

[local]
obsidian_vault = "/path/to/your/vault"  # Optional

[logging]
level = "info"  # Optional: trace, debug, info, warn, error
```

**Note**: Despite the section being named `[anthropic]`, the application now uses the `llm` crate which supports multiple backends. This naming is maintained for backward compatibility.

## Features

- **TUI Interface**: Clean, responsive terminal user interface
- **Multi-LLM Support**: Supports OpenAI, Anthropic, and other providers via the `llm` crate
- **Tool Integration**: Built-in tools for campaign management
- **Obsidian Integration**: Search and manage notes in Obsidian vaults
- **Conversation History**: Maintains chat history with context management
- **Error Handling**: Robust error handling with retry logic

## Building

```bash
cargo build --release
```

## Testing

```bash
cargo test
```

## Dependencies

The main dependencies include:
- `llm`: Unified LLM interface supporting multiple providers
- `rmcp`: Model Context Protocol for tool integration
- `ratatui`: Terminal user interface framework
- `tokio`: Async runtime
- `config`: Configuration management
- `serde`: Serialization/deserialization

## Architecture

- **Main application** (`src/main.rs`): Application entry point and event loop
- **LLM Client** (`src/llm_client.rs`): Unified LLM interface replacing the old Anthropic client
- **Event System** (`src/events.rs`): UI event handling
- **Tool Integration** (`src/obsidian.rs`): Obsidian vault tools
- **TUI** (`src/tui.rs`): Terminal user interface
- **Error Handling** (`src/errors.rs`): Centralized error types

## License

MIT