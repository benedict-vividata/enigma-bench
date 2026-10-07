#!/usr/bin/env bash
#
# Default-deny egress with a name-based allowlist.
#
# Modelled on the reference dev container Anthropic ships with Claude Code
# (https://github.com/anthropics/claude-code/blob/main/.devcontainer/init-firewall.sh).
# Runs as root from entrypoint.sh, before the agent user is ever started, and
# needs NET_ADMIN + NET_RAW on the container.
#
# The point is not to contain a determined attacker — it is to make the task
# genuinely offline, so a model is trained rather than downloaded, while leaving
# the agent's own provider reachable.
set -euo pipefail

ALLOWLIST="${ENIGMA_BENCH_ALLOWED_DOMAINS:-/etc/enigma-bench/allowed-domains.txt}"
SET_NAME=enigma-bench-allowed

log() { printf '[firewall] %s\n' "$*" >&2; }

if [[ ! -r "$ALLOWLIST" ]]; then
    log "no allowlist at $ALLOWLIST; refusing to configure a firewall"
    exit 1
fi

# Start from a clean slate so a container restart is idempotent.
iptables -F
iptables -X
# Do not flush the NAT table: Docker's embedded 127.0.0.11 resolver depends
# on Docker-managed NAT plumbing, and removing it breaks DNS inside the
# container before the allowlist can be resolved.
ipset destroy "$SET_NAME" 2>/dev/null || true
ipset create "$SET_NAME" hash:net

# Resolve the allowlist before the policy flips to DROP, or DNS resolution of
# the later entries would itself be blocked.
resolved=0
while read -r line; do
    domain="${line%%#*}"
    domain="$(echo "$domain" | tr -d '[:space:]')"
    [[ -z "$domain" ]] && continue
    # Prefer libc's resolver: Docker's embedded DNS can answer getent while
    # direct queries through dig are unavailable in some runtime setups.
    addresses="$(getent ahostsv4 "$domain" | awk '{print $1}' | sort -u || true)"
    if [[ -z "$addresses" ]]; then
        addresses="$(dig +short +time=3 +tries=2 A "$domain" | grep -E '^[0-9.]+$' || true)"
    fi
    if [[ -z "$addresses" ]]; then
        log "WARNING could not resolve $domain; the agent may fail to reach it"
        continue
    fi
    while read -r address; do
        [[ -z "$address" ]] && continue
        ipset add "$SET_NAME" "$address" 2>/dev/null || true
        resolved=$((resolved + 1))
    done <<< "$addresses"
    log "allow $domain -> $(echo "$addresses" | tr '\n' ' ')"
done < "$ALLOWLIST"

if (( resolved == 0 )); then
    log "resolved no addresses at all; refusing to lock the container out entirely"
    exit 1
fi

# Loopback and already-established flows.
iptables -A INPUT  -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A INPUT  -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# DNS to whatever resolver Docker handed us, so the CLIs can re-resolve.
while read -r nameserver; do
    iptables -A OUTPUT -p udp --dport 53 -d "$nameserver" -j ACCEPT
    iptables -A OUTPUT -p tcp --dport 53 -d "$nameserver" -j ACCEPT
done < <(awk '/^nameserver/ {print $2}' /etc/resolv.conf)

# The allowlist itself.
iptables -A OUTPUT -m set --match-set "$SET_NAME" dst -j ACCEPT

iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT DROP

# Prove both halves of the policy rather than assuming them.
if curl --silent --max-time 5 https://example.com >/dev/null 2>&1; then
    log "FAILED: example.com is still reachable; the deny rule is not effective"
    exit 1
fi
if ! curl --silent --max-time 10 --fail-with-body https://api.anthropic.com/v1/models >/dev/null 2>&1; then
    # A 401 is a success here: it means the TCP+TLS path is open.
    if ! curl --silent --max-time 10 --output /dev/null --write-out '%{http_code}' \
        https://api.anthropic.com/v1/models | grep -qE '^[0-9]{3}$'; then
        log "WARNING api.anthropic.com is not reachable; a Claude Code run will fail"
    fi
fi

log "egress restricted to $resolved address(es) from $(basename "$ALLOWLIST")"
