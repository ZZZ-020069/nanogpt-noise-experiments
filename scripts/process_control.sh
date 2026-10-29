#!/usr/bin/env bash
# Source in a supervising shell. Keep the supervisor alive until its child exits.
CONTROL_CHILD=''
CONTROL_STOP=0
control_stop() {
  CONTROL_STOP=1
  if [[ -n "$CONTROL_CHILD" ]]; then
    kill -TERM "$CONTROL_CHILD" 2>/dev/null || true
  fi
}
trap control_stop TERM INT

run_child() {
  local result=0
  (( CONTROL_STOP == 0 )) || return 75
  "$@" &
  CONTROL_CHILD=$!
  # Handle a stop arriving between launch and registration of the child PID.
  if (( CONTROL_STOP )); then control_stop; fi
  while true; do
    if wait "$CONTROL_CHILD"; then result=0; else result=$?; fi
    # A trapped signal interrupts Bash wait before the child has saved/exited.
    if kill -0 "$CONTROL_CHILD" 2>/dev/null; then continue; fi
    break
  done
  CONTROL_CHILD=''
  if (( CONTROL_STOP )); then return 75; fi
  return "$result"
}
