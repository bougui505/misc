#!/bin/bash
# push-pico.sh - Script to push main.py to Raspberry Pi Pico using mpremote and reset it

echo "Pushing main.py to Pico using mpremote..."

# Copy local main.py to the Pico
apptainer exec --writable-tmpfs ampy.sif mpremote cp main.py :main.py

if [ $? -eq 0 ]; then
    echo "Success! main.py has been pushed to the Pico."
    echo "Soft-resetting the Pico to run the new code..."
    apptainer exec --writable-tmpfs ampy.sif mpremote reset
    if [ $? -eq 0 ]; then
        echo "Pico has been successfully reset and is running the new code."
    else
        echo "Warning: Failed to reset the Pico automatically."
    fi
else
    echo "Error: Failed to push main.py to the Pico."
    exit 1
fi
