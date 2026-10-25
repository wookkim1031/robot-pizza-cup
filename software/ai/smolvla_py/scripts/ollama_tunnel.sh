#!/usr/bin/env bash
# Forward localhost:11434 to Ollama in gpu1's Jupyter container, for run_agent.py --model ollama:<name>.
# Starts Ollama there first if it is not running (~/ollama-serve.sh, models in ~/.ollama).
# run_agent.py and the web UI call this themselves when a local model is picked and the server is not up.
#   scripts/ollama_tunnel.sh              # gpu1 (nebius@81.85.4.50); GPU_HOST=nebius@<ip> to override
set -euo pipefail
HOST=${GPU_HOST:-nebius@81.85.4.50}
SSH=(ssh -i ~/.ssh/id_ed25519 -o ConnectTimeout=10 "$HOST")
if curl -s -m 2 localhost:11434/api/version >/dev/null; then echo "tunnel already up"; exit 0; fi
CID=$("${SSH[@]}" "sudo docker ps --format '{{.ID}} {{.Image}}' | grep -i jupyter | cut -d' ' -f1")
IP=$("${SSH[@]}" "sudo docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $CID")
"${SSH[@]}" "sudo docker exec -u jovyan $CID bash -l /home/jovyan/ollama-serve.sh"
pkill -f -- "-L 11434:" || true   # a dead tunnel would still hold the port
# Detached from our stdio, or a caller reading our output (run_agent.py, the web UI) would wait on it forever.
"${SSH[@]}" -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L "11434:$IP:11434" \
    </dev/null >/dev/null 2>>"${TMPDIR:-/tmp}/ollama_tunnel.log"
sleep 1
curl -s -m 5 localhost:11434/api/version && echo "  <- Ollama on $HOST ($IP) at http://localhost:11434"
