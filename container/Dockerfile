# syntax=docker/dockerfile:1
#
# Enigma Bench task container.
#
# Ubuntu 24.04 + uv + Node + the Claude Code and Codex CLIs. No CUDA base image
# is needed: the PyTorch wheels bundle their own CUDA runtime, and the NVIDIA
# Container Toolkit injects the driver at `docker run --gpus all`.
#
# Two unprivileged users:
#   agent  uid 2001  runs the coding agent; owns /workspace
#   bench  uid 2000  owns the held-out corpus and the run log; runs the MCP server
#
# The agent reaches the scorer only through `sudo -u bench` on one root-owned
# wrapper script, so it can be scored without ever reading the test corpus or
# the run log.
FROM ubuntu:24.04

ARG NODE_MAJOR=22
ARG CLAUDE_CODE_VERSION=latest
ARG CODEX_VERSION=latest
# 'hf:<dataset>:<config>' at build time, or 'dir:/mnt/corpus' with a bind mount.
ARG CORPUS_SOURCE=hf:Salesforce/wikitext:wikitext-103-raw-v1
ARG CORPUS_TRAIN_MIB=96

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_XET=1

# build-essential + python3-dev: torch.compile is part of the task surface, and
# Triton/Inductor JIT-compile kernels with the host C toolchain at run time,
# including a CPython extension (cuda_utils.c) that needs Python.h — the venv
# runs on the system Python 3.12, whose headers only ship in python3-dev.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git jq less sudo tini vim procps htop \
        iproute2 iptables ipset dnsutils build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Node, for the two agent CLIs.
RUN curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g \
        "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
        "@openai/codex@${CODEX_VERSION}" \
    && npm cache clean --force

COPY --from=ghcr.io/astral-sh/uv:0.11.23 /uv /uvx /usr/local/bin/

RUN groupadd --gid 2000 bench \
    && useradd --uid 2000 --gid 2000 --create-home --shell /bin/bash bench \
    && groupadd --gid 2001 agent \
    && useradd --uid 2001 --gid 2001 --create-home --shell /bin/bash agent

# --------------------------------------------------------------------------- #
# The benchmark package, into a root-owned virtualenv everyone can execute.
# --------------------------------------------------------------------------- #
ENV UV_PROJECT_ENVIRONMENT=/opt/enigma-bench/.venv \
    UV_LINK_MODE=copy \
    PATH=/opt/enigma-bench/.venv/bin:$PATH

WORKDIR /opt/enigma-bench
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts
COPY corpus ./host-corpus
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra agent \
    && chmod -R a+rX /opt/enigma-bench

# --------------------------------------------------------------------------- #
# Corpus, baked at build time: the container has no internet at run time.
# --------------------------------------------------------------------------- #
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f /opt/enigma-bench/host-corpus/manifest.json ]; then \
        install -d /srv/enigma-bench/corpus \
        && cp /opt/enigma-bench/host-corpus/train.txt \
              /opt/enigma-bench/host-corpus/valid.txt \
              /opt/enigma-bench/host-corpus/test.txt \
              /opt/enigma-bench/host-corpus/manifest.json \
              /srv/enigma-bench/corpus/; \
    else \
        uv run --no-project --python 3.12 --with "datasets>=4,<6" \
            python scripts/fetch_corpus.py \
            --out /srv/enigma-bench/corpus \
            --source "${CORPUS_SOURCE}" \
            --train-mib "${CORPUS_TRAIN_MIB}"; \
    fi \
    && chown -R root:root /srv/enigma-bench/corpus \
    && chmod 0755 /srv/enigma-bench/corpus \
    && chmod 0644 /srv/enigma-bench/corpus/train.txt \
                  /srv/enigma-bench/corpus/valid.txt \
                  /srv/enigma-bench/corpus/manifest.json \
    && chown bench:bench /srv/enigma-bench/corpus/test.txt \
    && chmod 0600 /srv/enigma-bench/corpus/test.txt

# Run log: bench only. Submissions: agent writes, bench reads (setgid).
RUN install -d -o bench -g bench -m 0700 /srv/enigma-bench/runs \
    && install -d -o agent -g bench -m 2750 /srv/enigma-bench/submissions \
    && install -d -o root  -g root  -m 0755 /srv/enigma-bench/out

# --------------------------------------------------------------------------- #
# The agent's workspace and its view of the task.
# --------------------------------------------------------------------------- #
COPY task/ /workspace/
COPY --chmod=0755 container/entrypoint.sh /opt/enigma-bench/bin/entrypoint.sh
COPY --chmod=0755 container/run-agent.sh /opt/enigma-bench/bin/run-agent.sh
COPY --chmod=0755 container/reset-agent-state.sh /opt/enigma-bench/bin/reset-agent-state.sh
COPY --chmod=0755 container/enigma-bench-mcp /opt/enigma-bench/bin/enigma-bench-mcp
COPY --chmod=0755 container/init-firewall.sh /opt/enigma-bench/bin/init-firewall.sh
COPY --chmod=0755 container/bench-cli /opt/enigma-bench/bin/bench-cli
COPY container/allowed-domains.txt /etc/enigma-bench/allowed-domains.txt

# The agent may run exactly one command as `bench`: the MCP wrapper. The wrapper
# and its interpreter are root-owned and not writable by `agent`, so this grants
# no path to arbitrary code as `bench`.
RUN printf '%s\n' \
        'agent ALL=(bench) NOPASSWD: /opt/enigma-bench/bin/enigma-bench-mcp' \
        > /etc/sudoers.d/enigma-bench \
    && chmod 0440 /etc/sudoers.d/enigma-bench \
    && visudo -cf /etc/sudoers.d/enigma-bench

RUN chown -R agent:agent /workspace \
    && install -d -o agent -g agent -m 0755 /home/agent/.claude /home/agent/.codex \
    && cp /workspace/codex-config.toml /home/agent/.codex/config.toml \
    && chown agent:agent /home/agent/.codex/config.toml

ENV ENIGMA_BENCH_HOME=/srv/enigma-bench \
    CLAUDE_CONFIG_DIR=/home/agent/.claude \
    CODEX_HOME=/home/agent/.codex \
    DISABLE_AUTOUPDATER=1 \
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
    ENIGMA_BENCH_HOURS=6 \
    ENIGMA_BENCH_SIZE=large \
    ENIGMA_BENCH_AGENT=claude

WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/enigma-bench/bin/entrypoint.sh"]
CMD ["run"]
