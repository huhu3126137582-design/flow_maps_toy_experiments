#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${LMD_TMUX_SESSION:-flow-lmd}"
OUTPUT_DIR="$ROOT_DIR/outputs/lmd"
PYTHON_BIN="${FLOW_MAPS_PYTHON:-python}"

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed or is not on PATH." >&2
    exit 1
fi
if ! PYTHON_BIN="$(command -v "$PYTHON_BIN")"; then
    echo "Python executable not found. Set FLOW_MAPS_PYTHON." >&2
    exit 1
fi

arguments=("$@")
for ((index = 0; index < ${#arguments[@]}; index++)); do
    case "${arguments[$index]}" in
        --output-dir)
            if ((index + 1 >= ${#arguments[@]})); then
                echo "--output-dir requires a value." >&2
                exit 1
            fi
            ((index += 1))
            OUTPUT_DIR="${arguments[$index]}"
            ;;
        --output-dir=*)
            OUTPUT_DIR="${arguments[$index]#--output-dir=}"
            ;;
    esac
done
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$ROOT_DIR/$OUTPUT_DIR"
fi

printf -v root_q "%q" "$ROOT_DIR"
printf -v python_q "%q" "$PYTHON_BIN"
printf -v pilot_summary_q "%q" "$OUTPUT_DIR/pilot_summary.json"
extra_arguments=""
for argument in "$@"; do
    printf -v argument_q "%q" "$argument"
    extra_arguments+=" $argument_q"
done

train_command="set -eo pipefail; cd $root_q"
train_command+=" && if [[ ! -f $pilot_summary_q ]]; then"
train_command+=" $python_q -u lmd_code/distill_lmd.py$extra_arguments"
train_command+=" --steps 2000 --resume auto --pilot-check;"
train_command+=" fi"
train_command+=" && grep -q '\"passed\": true' $pilot_summary_q"
train_command+=" && $python_q -u lmd_code/distill_lmd.py$extra_arguments --resume auto"

if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux new-session -d -s "$SESSION_NAME" bash -lc "$train_command"
    tmux set-option -t "$SESSION_NAME" history-limit 100000
    echo "Started tmux session: $SESSION_NAME"
else
    echo "Attaching existing tmux session: $SESSION_NAME"
fi

if [[ ! -t 0 || ! -t 1 ]]; then
    echo "Attach with: tmux attach-session -t $SESSION_NAME"
    exit 0
fi

if [[ -n "${TMUX:-}" ]]; then
    exec tmux switch-client -t "$SESSION_NAME"
fi
exec tmux attach-session -t "$SESSION_NAME"
