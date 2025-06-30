# dmcli

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](#)  
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

### Installation

#### Option 1: Install from Release

1. Download the latest release from the [Releases page](../../releases)
2. Extract the archive:
   ```bash
   tar -xzf dmcli2-x86_64-unknown-linux-gnu.tar.gz
   ```
3. Move the binary to your PATH:
   ```bash
   sudo mv dmcli2 /usr/local/bin/
   ```

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

3. The binary will be available at `target/release/dmcli2`

### Usage

Run the application:
```bash
dmcli2
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