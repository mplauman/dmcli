FROM ubuntu:latest

# v1.2.27/opencode-linux-x64.tar.gz
# v1.2.27/opencode-linux-x64-musl.tar.gz
ARG opencode=v1.2.27/opencode-linux-x64.tar.gz

# 2.1.110/linux-x64-musl/claude
ARG claude=2.1.110/linux-x64/claude

RUN apt update && \
    apt install -y curl zstd && \
    echo "Downloading opencode..." && \
    curl -fsSL https://github.com/anomalyco/opencode/releases/download/$opencode | gunzip -c | tar -xf - -C /usr/local/bin && \
    echo "Downloading claude..." && \
    curl -fsSL https://storage.googleapis.com/claude-code-dist-86c565f3-f756-42ad-8dfa-d59b1c096819/claude-code-releases/$claude > /usr/local/bin/claude && chmod +x /usr/local/bin/claude && \
    echo "Done"

# Create a non-root user with home directory
RUN useradd -m -s /bin/bash user

# Set the user as the default
USER user
