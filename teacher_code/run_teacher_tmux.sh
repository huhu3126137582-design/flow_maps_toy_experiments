#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${TEACHER_TMUX_SESSION:-flow-teacher}"
PYTHON_BIN="${FLOW_MAPS_PYTHON:-python}"

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed or is not on PATH." >&2
    exit 1
fi
if ! PYTHON_BIN="$(command -v "$PYTHON_BIN")"; then
    echo "Python executable not found. Set FLOW_MAPS_PYTHON." >&2
    exit 1
fi

printf -v root_q "%q" "$ROOT_DIR"
printf -v python_q "%q" "$PYTHON_BIN"
train_command="cd $root_q"
train_command+=" && $python_q -u teacher_code/train_teacher.py --resume auto"
for argument in "$@"; do
    printf -v argument_q "%q" "$argument"
    train_command+=" $argument_q"
done

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
