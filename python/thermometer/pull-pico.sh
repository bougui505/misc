#!/bin/bash
# pull-pico.sh - Script to pull main.py from Raspberry Pi Pico using mpremote

echo "Pulling main.py using mpremote..."

# Run mpremote via apptainer. mpremote auto-detects the Pico port automatically.
apptainer exec --writable-tmpfs ampy.sif mpremote cp :main.py main.py.backup

if [ $? -eq 0 ] && [ -s main.py.backup ]; then
    mv main.py.backup main.py
    echo "Success! main.py has been pulled and saved successfully."
else
    echo "Error: Failed to retrieve main.py from the Pico."
    rm -f main.py.backup
    exit 1
fi
