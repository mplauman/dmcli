#!/bin/bash
set -euo pipefail

# Change to the project's root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up git hooks for dmcli2..."

# Ensure the hooks/ directory exists
if [ ! -d "hooks" ]; then
    echo "Error: hooks directory not found!"
    echo "Make sure you're running this script from the project root directory."
    exit 1
fi

# Create .git/hooks directory if it doesn't exist
mkdir -p .git/hooks

# Install pre-commit hook
echo "Installing pre-commit hook..."
cp hooks/pre-commit .git/hooks/
chmod +x .git/hooks/pre-commit

# Ensure Rust components are installed
echo "Checking Rust components..."

# Check for rustfmt
if ! rustup component list | grep -q "rustfmt (installed)"; then
    echo "Installing rustfmt component..."
    rustup component add rustfmt
else
    echo "rustfmt is already installed."
fi

# Check for clippy
if ! rustup component list | grep -q "clippy (installed)"; then
    echo "Installing clippy component..."
    rustup component add clippy
else
    echo "clippy is already installed."
fi

echo "Git hooks setup complete!"
echo "The following hooks are now installed:"
echo "  - pre-commit: Runs rustfmt and clippy"
echo ""
echo "You can customize these hooks by editing the files in the 'hooks/' directory"
echo "and then running this setup script again."