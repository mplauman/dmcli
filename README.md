# dmcli

[![CI](https://github.com/mplauman/dmcli/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mplauman/dmcli/actions/workflows/ci.yml)  
[![Build and Release](https://github.com/mplauman/dmcli/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/mplauman/dmcli/actions/workflows/build.yml)  
[![Security Audit](https://github.com/mplauman/dmcli/actions/workflows/security.yml/badge.svg?branch=main)](https://github.com/mplauman/dmcli/actions/workflows/security.yml)  
[![Language](https://img.shields.io/badge/language-Rust-orange)](#)  
[![License](https://img.shields.io/badge/license-MIT-blue)](#)

## Overview

`dmcli` is a Rust-based Dungeon Master's helper tool designed to search through Obsidian notes for session and world building information. It provides a command-line interface to help DMs quickly find and reference their campaign notes during gameplay.

## Features

- Written in Rust for high performance and reliability
- **Integrated dice roller** using the [caith](https://crates.io/crates/caith) library for comprehensive dice notation support
- **LLM integration** for natural language searches through Obsidian vaults
- **Sliding window chat memory** with configurable strategies for conversation management
- **Conversation search functionality** with keyboard shortcuts for searching chat history
- **Obsidian vault integration** for searching and accessing campaign notes
- Terminal-based user interface for quick access during gameplay
- Configurable settings via TOML configuration file

## Getting Started

### Prerequisites

- Rust 1.70+ (2024 edition)
- Git

### Supported Platforms

`dmcli` supports the following platforms:
- **Linux**: x86_64-unknown-linux-gnu
- **macOS**: x86_64-apple-darwin (Intel), aarch64-apple-darwin (Apple Silicon)
- **Windows**: x86_64-pc-windows-msvc

### Installation

#### Option 1: Install from Release

1. Download the latest release for your platform from the [Releases page](../../releases)
2. Extract the archive:
   
   **Linux/macOS:**
   ```bash
   tar -xzf dmcli-<target>.tar.gz
   ```
   
   **Windows:**
   ```powershell
   Expand-Archive -Path dmcli-<target>.zip -DestinationPath .
   ```

3. Move the binary to your PATH:
   
   **Linux/macOS:**
   ```bash
   sudo mv dmcli /usr/local/bin/
   ```
   
   **Windows:**
   Move `dmcli.exe` to a directory in your PATH or add the current directory to your PATH environment variable.

#### Option 2: Compile from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/mplauman/dmcli.git
   cd dmcli
   ```

2. Build the project:
   ```bash
   cargo build --release
   ```

3. The binary will be available at:
   - **Linux/macOS:** `target/release/dmcli`
   - **Windows:** `target/release/dmcli.exe`

### Usage

Run the application:
```bash
dmcli
```

Use the built-in commands:
- `roll 2d6` - Roll dice using standard dice notation
- `exit` - Exit the application

### Keyboard Shortcuts

- **Ctrl+F** - Enter search mode to search through conversation history
- **Ctrl+G** - Navigate to next search result
- **Ctrl+Shift+G** - Navigate to previous search result
- **Esc** - Exit search mode
- **PgUp/PgDn** - Scroll through conversation history

## Configuration

`dmcli` uses a TOML configuration file located at `~/.config/dmcli.toml` (or your system's equivalent config directory).

### Configuration Options

Create a `dmcli.toml` file with the following structure:

```toml
[anthropic]
# API key for Anthropic Claude
api_key = "your-api-key-here"

# Model to use (default: claude-3-5-haiku-20241022)
model = "claude-3-5-haiku-20241022"

# Maximum tokens for responses (default: 8192)
max_tokens = 8192

# Sliding window size for chat memory management (default: 32)
# This determines how many recent messages are kept in memory
# before older messages are trimmed using the configured strategy
window_size = 32

# Trim strategy for sliding window memory management (default: "summarize")
# Available options:
# - "summarize": Summarizes older messages when the window is full
# - "drop": Simply drops older messages when the window is full
trim_strategy = "summarize"

[local]
# Path to your Obsidian vault for note searching
obsidian_vault = "/path/to/your/obsidian/vault"

[logging]
# Enable OpenTelemetry logging (optional)
opentelemetry = false
```

### Environment Variables

Configuration can also be set using environment variables with the `DMCLI_` prefix:

```bash
export DMCLI_ANTHROPIC_API_KEY="your-api-key-here"
export DMCLI_ANTHROPIC_MODEL="claude-3-5-haiku-20241022"
export DMCLI_ANTHROPIC_MAX_TOKENS=8192
export DMCLI_ANTHROPIC_WINDOW_SIZE=32
export DMCLI_ANTHROPIC_TRIM_STRATEGY="summarize"
export DMCLI_LOCAL_OBSIDIAN_VAULT="/path/to/your/vault"
export DMCLI_LOGGING_OPENTELEMETRY=false
```

## Chat Memory Management

`dmcli` features an intelligent sliding window system for managing chat history:

### Sliding Window
- **Window Size**: Configurable number of recent messages to keep in memory (default: 32)
- **Automatic Trimming**: Older messages are automatically managed when the window is full
- **Context Preservation**: Important system messages and context are preserved

### Trim Strategies
- **Summarize** (default): Uses AI to summarize older messages, preserving important context while reducing memory usage
- **Drop**: Simply removes older messages when the window is full (faster but may lose context)

### Conversation Search
- **Real-time Search**: Search through your entire conversation history with Ctrl+F
- **Context Snippets**: Search results show relevant context around matches
- **Navigation**: Use Ctrl+G and Ctrl+Shift+G to navigate between search results
- **Visual Indicators**: Search mode is clearly indicated in the interface

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.