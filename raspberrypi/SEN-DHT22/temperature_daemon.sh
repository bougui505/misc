#!/usr/bin/env zsh

RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

DIRSCRIPT="$(dirname "$(readlink -f "$0")")"

tmux has-session -t "T" 2>/dev/null

if [ $? != 0 ]; then
    echo "${RED}temperature monitoring starting in tmux session T${NC}"
    tmux new-session -d -s "T" "$DIRSCRIPT/temperature_sampler.sh"
else
    echo "${CYAN}temperature monitoring running on tmux session: 'T', run the following command to attach${NC}"
    echo "${GREEN}tmux attach -t 'T'${NC}"
fi
