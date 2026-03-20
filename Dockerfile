FROM ubuntu:latest

# v1.17.0/qdrant-x86_64-unknown-linux-gnu.tar.gz
# v1.17.0/qdrant-x86_64-unknown-linux-musl.tar.gz
ARG qdrant=v1.17.0/qdrant-x86_64-unknown-linux-gnu.tar.gz

# v1.2.27/opencode-linux-x64.tar.gz
# v1.2.27/opencode-linux-x64-musl.tar.gz
ARG opencode=v1.2.27/opencode-linux-x64.tar.gz

RUN apt update && \
    apt install -y curl && \
    curl -fsSL https://github.com/qdrant/qdrant/releases/download/$qdrant | gunzip -c | tar -xf - -C /usr/local/bin && \
    curl -fsSL https://github.com/anomalyco/opencode/releases/download/$opencode | gunzip -c | tar -xf - -C /usr/local/bin
