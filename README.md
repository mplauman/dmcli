# dmcli

[![CI](https://github.com/mplauman/dmcli/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mplauman/dmcli/actions/workflows/ci.yml)  
[![Build and Release](https://github.com/mplauman/dmcli/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/mplauman/dmcli/actions/workflows/build.yml)  
[![Security Audit](https://github.com/mplauman/dmcli/actions/workflows/security.yml/badge.svg?branch=main)](https://github.com/mplauman/dmcli/actions/workflows/security.yml)  
[![Language](https://img.shields.io/badge/language-Rust-orange)](#)  
[![License](https://img.shields.io/badge/license-MIT-blue)](#)

## Overview

`dmcli` is a Dungeon Master's command-line toolkit. It provides dice rolling and
semantic search over Markdown notes using local BERT embeddings — no external AI
service required.

## Features

- **Dice roller** — full dice notation support via the [caith](https://crates.io/crates/caith) library, including repeated rolls and reason annotations
- **Semantic search** — hybrid dense + sparse vector search over any directory of Markdown files using a local `sentence-transformers/all-MiniLM-L6-v2` BERT model
- **Markdown indexing** — walks a directory recursively, splits files into overlapping chunks that respect heading boundaries, and stores them in a vector database
- **Qdrant backend** — optional [Qdrant](https://qdrant.tech) vector store with hybrid Reciprocal Rank Fusion (RRF) search; falls back to a no-op store when no URL is provided

## Getting Started

### Prerequisites

- Rust (2024 edition)
- Git
- A running [Qdrant](https://qdrant.tech) instance (optional; required for `index` and `search`)

### Supported Platforms

- **Linux**: x86_64-unknown-linux-gnu
- **macOS**: x86_64-apple-darwin (Intel), aarch64-apple-darwin (Apple Silicon)
- **Windows**: x86_64-pc-windows-msvc

### Installation

#### Option 1: Install from Release

1. Download the latest release for your platform from the [Releases page](../../releases).
2. Extract the archive:

   **Linux/macOS:**
   ```
   tar -xzf dmcli-<target>.tar.gz
   ```

   **Windows:**
   ```
   Expand-Archive -Path dmcli-<target>.zip -DestinationPath .
   ```

3. Move the binary to your PATH:

   **Linux/macOS:**
   ```
   sudo mv dmcli /usr/local/bin/
   ```

   **Windows:** Move `dmcli.exe` to any directory already on your PATH.

#### Option 2: Compile from Source

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

| Option | Description |
|---|---|
| `-q, --qdrant-url <URL>` | URL of the Qdrant gRPC endpoint (e.g. `http://localhost:6334`). Required for `index` and `search`. |

### Commands

#### `roll` — Roll dice

Roll any expression supported by the [caith syntax](https://github.com/Geobert/caith?tab=readme-ov-file#syntax).

```
dmcli roll <EXPR>
```

Examples:

```
dmcli roll 1d20
dmcli roll 2d6+3
dmcli roll 4d6 ^ 3        # roll 4d6, keep highest 3
dmcli roll 2d10 ! Stealth  # roll with a reason annotation
```

Single rolls print the total; repeated rolls print each result as a list.

#### `index` — Index a directory

Walk a directory recursively and embed all Markdown (`.md`) files into the
configured Qdrant collection. Hidden directories (names beginning with `.`)
are skipped. The collection is created automatically on first use.

Model weights are downloaded from Hugging Face on first run and cached locally
by `hf-hub`.

```
dmcli --qdrant-url http://localhost:6334 index /path/to/notes
```

#### `search` — Search indexed notes

Run a semantic query against previously indexed content. Results are ranked by
hybrid dense + sparse similarity and printed with their source file, heading
breadcrumb, relevance score, and matching text.

```
dmcli --qdrant-url http://localhost:6334 search what resistances do fire giants have
```

Output format per result:

```
Source:  /path/to/notes/monsters.md#Monsters/Fire Giant
Score:   0.8731
Fire giants are immune to fire damage and resistant to ...
```

## Architecture

`dmcli` is a Cargo workspace with two crates:

- **`lib`** — core library: dice rolling, document indexing, embedding, and vector store abstractions
- **`cli`** — thin binary that wires `lib` to `clap`-parsed subcommands

### Embedding & Search Pipeline

1. **Chunking** — Markdown files are parsed with `pulldown-cmark`. Heading
   boundaries are preserved and used as section metadata. Each logical section
   is further split into overlapping token-budget chunks (~256 tokens, 50-token
   overlap) by `text-splitter`.
2. **Embedding** — Each chunk is encoded into a 384-dimensional dense vector
   using the `all-MiniLM-L6-v2` BERT model running locally via `candle`.
   A sparse log-TF vector is also computed from the tokenised text.
3. **Storage** — Chunks are upserted into Qdrant under a `dmcli_chunks`
   collection with named `dense` (Cosine HNSW) and `sparse` vector fields.
   Upserts are idempotent — each chunk is keyed by the SHA-256 of its text.
4. **Retrieval** — Queries are encoded the same way. Keywords are extracted
   with a BERT-based n-gram scoring pass. Dense and sparse prefetch queries
   are issued and fused with Reciprocal Rank Fusion (RRF).

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.