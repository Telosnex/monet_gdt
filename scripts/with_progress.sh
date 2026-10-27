#!/bin/sh
# Run a command; every 10 s print what it is, elapsed time, and the given
# estimate minus elapsed. Usage: with_progress.sh "<what>" <estimate_s> <cmd...>
what="$1"; est="$2"; shift 2
log=$(mktemp); start=$(date +%s)
"$@" > "$log" 2>&1 & pid=$!
echo "[$(date +%H:%M:%S)] start $what (estimate ${est}s)"
while kill -0 $pid 2>/dev/null; do
  sleep 10; kill -0 $pid 2>/dev/null || break
  el=$(( $(date +%s) - start )); left=$(( est - el ))
  [ $left -lt 0 ] && left="unknown (past the ${est}s estimate)" || left="~${left}s"
  echo "[$(date +%H:%M:%S)] .. $what: elapsed ${el}s, remaining $left; last output: $(tail -c 160 "$log" | tr '\r\n' '  ' | tail -c 120)"
done
wait $pid; rc=$?
echo "[$(date +%H:%M:%S)] done $what: exit $rc, elapsed $(( $(date +%s) - start ))s"; tail -5 "$log"; rm -f "$log"; exit $rc
