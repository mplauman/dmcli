# dmcli

[![CI](https://github.com/mplauman/dmcli/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mplauman/dmcli/actions/workflows/ci.yml)
[![Build and Release](https://github.com/mplauman/dmcli/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/mplauman/dmcli/actions/workflows/build.yml)
[![Security Audit](https://github.com/mplauman/dmcli/actions/workflows/security.yml/badge.svg?branch=main)](https://github.com/mplauman/dmcli/actions/workflows/security.yml)
[![Language](https://img.shields.io/badge/language-Rust-orange)](#)
[![License](https://img.shields.io/badge/license-MIT-blue)](#)

## Overview

`dmcli` is a Dungeon Master's command-line toolkit. It provides dice rolling
via full dice notation powered by the [caith](https://crates.io/crates/caith)
library, with multiple output formats for human reading, LLM prompt injection,
and agent integrations.

## Features

- **Dice roller** — full dice notation via [caith](https://crates.io/crates/caith), including repeated rolls and reason annotations
- **Multiple output formats** — `--output markdown` (default), `--output xml` (for LLM prompt injection), and `--output json` (for agent integrations)

## Getting Started

### Prerequisites

- Rust 1.85+ (2024 edition)

### Supported Platforms

| Platform | Target |
|---|---|
| Linux (x86_64) | `x86_64-unknown-linux-gnu` |
| macOS (Intel) | `x86_64-apple-darwin` |
| macOS (Apple Silicon) | `aarch64-apple-darwin` |
| Windows (x86_64) | `x86_64-pc-windows-msvc` |

### Installation

#### Option 1: Download a Release

1. Download the archive for your platform from the [Releases page](../../releases).
2. Extract it:

   **Linux / macOS**
   ```
   tar -xzf dmcli-<target>.tar.gz
   sudo mv dmcli /usr/local/bin/
   ```

   **Windows**
   ```
   Expand-Archive -Path dmcli-<target>.zip -DestinationPath .
   ```
   Move `dmcli.exe` to any directory on your `PATH`.

#### Option 2: Build from Source

```
git clone https://github.com/mplauman/dmcli.git
cd dmcli
cargo build --release
```

The binary will be at `target/release/dmcli` (or `target/release/dmcli.exe` on Windows).

## Usage

```
dmcli <COMMAND>
```

---

### `roll` — Roll dice

Roll any expression supported by the [caith syntax](https://github.com/Geobert/caith?tab=readme-ov-file#syntax).

```
dmcli roll [--output <FORMAT>] <EXPR>...
```

**Examples**

```sh
dmcli roll 1d20
dmcli roll 2d6+3
dmcli roll 4d6 ^ 3          # roll 4d6, keep highest 3
dmcli roll 2d10 ! Stealth   # reason annotation
dmcli roll --output json 1d20
```

**Output formats**

| Format | Single roll | Multi roll |
|---|---|---|
| `markdown` (default) | `🎲 **17**` | `🎲🎲 3, 5, 2 — total: **10**` |
| `xml` | `<roll type="single"><total>17</total></roll>` | `<roll type="multi"><values>3, 5, 2</values><total>10</total></roll>` |
| `json` | `{"type": "single", "total": 17}` | `{"type": "multi", "values": [3, 5, 2], "total": 10}` |

A reason annotation (e.g. `2d10 ! Stealth`) adds a `reason` field to the XML
and JSON output and a `*(reason)*` suffix to the Markdown output.

---

## Architecture

`dmcli` is a Cargo workspace with two crates:

- **`lib`** — core library: dice rolling and the three output serialisation formats
- **`cli`** — thin binary that parses subcommands with `clap` and delegates to `lib`

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
