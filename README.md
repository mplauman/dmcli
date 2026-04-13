# dmcli

[![CI](https://github.com/mplauman/dmcli/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mplauman/dmcli/actions/workflows/ci.yml)
[![Build and Release](https://github.com/mplauman/dmcli/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/mplauman/dmcli/actions/workflows/build.yml)
[![Security Audit](https://github.com/mplauman/dmcli/actions/workflows/security.yml/badge.svg?branch=main)](https://github.com/mplauman/dmcli/actions/workflows/security.yml)
[![Language](https://img.shields.io/badge/language-Rust-orange)](#)
[![License](https://img.shields.io/badge/license-MIT-blue)](#)

## Overview

`dmcli` is a Dungeon Master's command-line toolkit. It provides dice rolling and
semantic search over Markdown notes using local BERT embeddings — no external AI
service or internet connection required after the first run.

## Features

- **Dice roller** — full dice notation via the [caith](https://crates.io/crates/caith) library, including repeated rolls and reason annotations
- **Markdown indexing** — walks a directory recursively, splits files into overlapping chunks that respect heading boundaries, and stores embeddings in a local SQLite database
- **Semantic search** — dense vector search over indexed notes using a local `sentence-transformers/all-MiniLM-L6-v2` BERT model; no cloud service required
- **Multiple output formats** — every command supports `--output markdown` (default), `--output xml` (for LLM prompt injection), and `--output json` (for agent integrations)
- **TOML config file** — persistent defaults for the database path and index directory; managed with `dmcli config init` / `dmcli config update`

## Getting Started

### Prerequisites

- Rust 1.85+ (2024 edition)

Model weights (~90 MB) are downloaded from Hugging Face on the first `index` or
`search` run and cached locally by `hf-hub`. No account or API key is required.

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
dmcli [OPTIONS] <COMMAND>
```

### Global Options

| Option | Default | Description |
|---|---|---|
| `-C, --config <PATH>` | `$XDG_CONFIG_HOME/dmcli/config.toml` | Path to the TOML configuration file |
| `-d, --db-path <PATH>` | `$XDG_DATA_HOME/dmcli/chunks.db` | Path to the SQLite vector database |

The XDG defaults expand to `~/.config/dmcli/config.toml` and
`~/.local/share/dmcli/chunks.db` when the environment variables are not set.

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

| Format | Example |
|---|---|
| `markdown` (default) | `🎲 **17**` |
| `xml` | `<roll type="single"><total>17</total></roll>` |
| `json` | `{"type":"single","total":17}` |

---

### `index` — Index a directory

Walk a directory recursively and embed all Markdown (`.md`) files into the
local SQLite database. Hidden directories (names beginning with `.`) are
skipped. The database file is created automatically on first use.

```
dmcli index [PATH]
```

If `PATH` is omitted, `dmcli.index.path` from the config file is used.

**Examples**

```sh
dmcli index /path/to/campaign/notes
dmcli --db-path ~/rpg/notes.db index /path/to/notes
```

Re-indexing is idempotent — each chunk is keyed by the SHA-256 of its text, so
unchanged content is never re-embedded.

---

### `search` — Search indexed notes

Run a semantic query against previously indexed content. Results are ranked by
dense cosine similarity and printed with their source file, heading breadcrumb,
relevance score, and matching text.

```
dmcli search [--output <FORMAT>] <QUERY>...
```

**Examples**

```sh
dmcli search what resistances do fire giants have
dmcli search --output xml how does grappling work
dmcli search --output json potion of healing effects
```

**Output formats**

| Format | Description |
|---|---|
| `markdown` (default) | Numbered results with `## [N]` headings, separated by `---` |
| `xml` | `<document index="N">` blocks for direct injection into an LLM prompt |
| `json` | Array of `{rank, score, source, text}` objects for agent integrations |

---

### `config` — Manage the configuration file

#### `config init`

Write a default, fully commented-out configuration file to the XDG config path
(or the `--config` override). Fails if the file already exists unless `--force`
is given. Parent directories are created as needed.

```
dmcli config init [--force]
```

#### `config update`

Add any keys that exist in the current schema but are missing from an existing
config file. All existing content and formatting is preserved; new keys are
inserted alphabetically with their description comment. Falls back to `init` if
no file exists.

```
dmcli config update
```

**Example config file** (`~/.config/dmcli/config.toml`)

```toml
[dmcli]
# Path to the SQLite database file used for vector storage.
# Default: $XDG_DATA_HOME/dmcli/chunks.db (~/.local/share/dmcli/chunks.db)
# db_path =

[dmcli.index]
# Default directory to index when none is supplied on the command line.
# path =
```

---

## Architecture

`dmcli` is a Cargo workspace with two crates:

- **`lib`** — core library: dice rolling, document indexing, embedding, search, and the SQLite vector store
- **`cli`** — thin binary that parses subcommands with `clap`, loads the TOML config, and delegates to `lib`

### Embedding & Search Pipeline

1. **Chunking** — Markdown files are parsed with `pulldown-cmark`. Heading
   boundaries are preserved as section metadata. Each section is further split
   into overlapping token-budget chunks (~256 tokens, ~50-token overlap) by
   `text-splitter`.

2. **Embedding** — Each chunk is encoded into a 384-dimensional dense vector
   using `sentence-transformers/all-MiniLM-L6-v2` running locally via
   `candle`. A sparse log-TF keyword vector is also computed from the
   tokenised text.

3. **Storage** — Chunks are upserted into a SQLite database (via `libsql`)
   with the dense vector stored as a little-endian binary blob. Each chunk is
   keyed by the SHA-256 of its text, making re-indexing idempotent.

4. **Retrieval** — Queries are encoded the same way as documents. Dense
   cosine similarity is computed across all stored vectors and the top results
   are returned, ranked by score.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.