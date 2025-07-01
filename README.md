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
- **Cross-platform support** - Available for Linux, macOS, and Windows
- **Integrated dice roller** using the [caith](https://crates.io/crates/caith) library for comprehensive dice notation support
- **LLM integration** for natural language searches through Obsidian vaults
- **Obsidian vault integration** for searching and accessing campaign notes
- Terminal-based user interface for quick access during gameplay
- Configurable settings via TOML configuration file

## Getting Started

### Supported Platforms

`dmcli` supports the following platforms:

- **Linux** (x86_64) - Tested on Ubuntu, Debian, CentOS, and other distributions
- **macOS** (Intel x86_64 and Apple Silicon ARM64) - macOS 10.15 and later
- **Windows** (x86_64) - Windows 10 and later

### Prerequisites

- Rust 1.70+ (2024 edition)
- Git

### Installation

#### Option 1: Install from Release

1. Download the latest release for your platform from the [Releases page](../../releases):
   - **Linux**: `dmcli-x86_64-unknown-linux-gnu.tar.gz`
   - **macOS (Intel)**: `dmcli-x86_64-apple-darwin.tar.gz`
   - **macOS (Apple Silicon)**: `dmcli-aarch64-apple-darwin.tar.gz`
   - **Windows**: `dmcli-x86_64-pc-windows-msvc.zip`

2. Extract the archive:
   
   **Linux/macOS:**
   ```bash
   tar -xzf dmcli-<your-platform>.tar.gz
   ```
   
   **Windows:**
   Extract the zip file using Windows Explorer or PowerShell:
   ```powershell
   Expand-Archive -Path dmcli-x86_64-pc-windows-msvc.zip -DestinationPath .
   ```

3. Move the binary to your PATH:
   
   **Linux/macOS:**
   ```bash
   sudo mv dmcli /usr/local/bin/
   ```
   
   **Windows:**
   Move `dmcli.exe` to a directory in your PATH, or add the current directory to your PATH.

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

3. The binary will be available at `target/release/dmcli`

### Usage

Run the application:
```bash
dmcli
```

Use the built-in commands:
- `roll 2d6` - Roll dice using standard dice notation
- `exit` - Exit the application

## Configuration

`dmcli` uses a TOML configuration file located at:

- **Linux/macOS**: `~/.config/dmcli.toml`
- **Windows**: `%APPDATA%\dmcli\dmcli.toml`

The application automatically detects your system's standard configuration directory.

### Configuration Options

Create a `dmcli.toml` file with the following structure:

```toml
[local]
# Path to your Obsidian vault for note searching
obsidian_vault = "/path/to/your/obsidian/vault"

[logging]
# Enable OpenTelemetry logging (optional)
opentelemetry = false
# Enable syslog logging (Unix/Linux/macOS only)
syslog = false
```

**Note**: The `syslog` option is only available on Unix-like systems (Linux and macOS). It will be ignored on Windows.

### Environment Variables

Configuration can also be set using environment variables with the `DMCLI_` prefix:

**Linux/macOS:**
```bash
export DMCLI_LOCAL_OBSIDIAN_VAULT="/path/to/your/vault"
export DMCLI_LOGGING_OPENTELEMETRY=false
```

**Windows (Command Prompt):**
```cmd
set DMCLI_LOCAL_OBSIDIAN_VAULT=C:\path\to\your\vault
set DMCLI_LOGGING_OPENTELEMETRY=false
```

**Windows (PowerShell):**
```powershell
$env:DMCLI_LOCAL_OBSIDIAN_VAULT="C:\path\to\your\vault"
$env:DMCLI_LOGGING_OPENTELEMETRY="false"
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.