#!/bin/bash
# Run this after studio restart
curl -fsSL https://ollama.com/install.sh | sh
export OLLAMA_MODELS=/teamspace/studios/this_studio/.ollama/models
ollama serve &
sleep 5
ollama list
echo "Ollama ready."
