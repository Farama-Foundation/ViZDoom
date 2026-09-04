FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        procps \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

# Used by the reference solution. Codex and Claude Code call the MCP server
# directly through the integrations Harbor installs for them.
RUN pip install --no-cache-dir fastmcp==2.14.5 numpy==2.3.2 pillow==11.3.0

WORKDIR /app
