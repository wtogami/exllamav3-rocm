#!/bin/bash
# usage: restart_engine.sh <abs_config> <abs_logfile>
# Stop the TabbyAPI engine running in the gg screen session (Ctrl-C to its
# foreground process group) and relaunch the run script, redirecting its
# stdout+stderr to a log file we can parse cleanly. Never attaches.
set -u
CFG="$1"; LOG="$2"
S=$(screen -ls | awk '/gg/{print $1; exit}')
[ -z "$S" ] && { echo "no screen session found"; exit 2; }
echo "session=$S  stopping engine (Ctrl-C)"
screen -S "$S" -X stuff $'\003'
for _ in $(seq 1 40); do pgrep -f "main.py --config" >/dev/null || break; sleep 1; done
if pgrep -f "main.py --config" >/dev/null; then
  echo "  still up after 40s; second Ctrl-C"; screen -S "$S" -X stuff $'\003'; sleep 6
fi
screen -S "$S" -X stuff $'\n'          # fresh prompt
: > "$LOG"
echo "  launching config=$CFG log=$LOG"
CMD="cd ~/exllamav3-rocm && rocm/scripts/run_tabbyapi.sh $CFG ../tabbyAPI > $LOG 2>&1"
screen -S "$S" -X stuff "$CMD"$'\n'   # literal command + Enter (no $() newline-stripping)
echo "  start command sent; not waiting for load here"
