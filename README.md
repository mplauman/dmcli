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

## Configuration

`dmcli` uses a TOML configuration file located at `~/.config/dmcli.toml` (or your system's equivalent config directory).

### Configuration Options

Create a `dmcli.toml` file with the following structure:

```toml
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
export DMCLI_LOCAL_OBSIDIAN_VAULT="/path/to/your/vault"
export DMCLI_LOGGING_OPENTELEMETRY=false
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.