#!/usr/bin/env bash

# Parse command line arguments
PORT=${1:-11435}  # Use first argument or default to 11435
export OLLAMA_HOST=127.0.0.1:${PORT}
export OLLAMA_CONTEXT_LENGTH=65536

echo "Starting Ollama server on ${OLLAMA_HOST} in the background..."

# 2. Start the server in the background using '&'.
# 'exec' is sometimes used to replace the current shell process,
# but simply running in the background with '&' is safer for this workflow.
ollama serve &

# Save the Process ID (PID) of the background server process
OLLAMA_PID=$!

# 3. Wait a few seconds for the server to initialize fully.
echo "Waiting 5 seconds for server initialization..."
sleep 5

# 4. Now, run the pull command as a client.
MODEL="gemma"
echo "Pulling model ${MODEL} to ${OLLAMA_HOST}..."
ollama pull ${MODEL}

# 5. Bring the server process back to the foreground (optional, but necessary if
# you want the script to wait until the server is killed).
echo "Model pulled. Server is now running in the foreground. Press Ctrl+C to stop."
wait $OLLAMA_PID

# Note: The 'wait' command above will block until the background server is terminated.
