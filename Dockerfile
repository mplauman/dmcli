FROM ubuntu:latest

# v1.2.27/opencode-linux-x64.tar.gz
# v1.2.27/opencode-linux-x64-musl.tar.gz
ARG opencode=v1.2.27/opencode-linux-x64.tar.gz

# v0.18.2/ollama-linux-amd64-rocm.tar.zst
# v0.18.2/ollama-linux-amd64.tar.zst
# v0.18.2/ollama-linux-arm64-jetpack5.tar.zst
# v0.18.2/ollama-linux-arm64-jetpack6.tar.zst
# v0.18.2/ollama-linux-arm64.tar.zst
#ARG ollama=v0.18.2/ollama-linux-amd64.tar.zst

RUN apt update && \
    apt install -y curl zstd && \
    echo "Downloading external deps..." && \
    curl -fsSL https://github.com/anomalyco/opencode/releases/download/$opencode | gunzip -c | tar -xf - -C /usr/local/bin && \
#    curl -fsSL https://github.com/ollama/ollama/releases/download/$ollama | zstd -d | tar -xf - -C /usr/local && \
    echo "Done"

