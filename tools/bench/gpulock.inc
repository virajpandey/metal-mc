# The GPU lock (sourced by run_mc.sh and gpuwait.sh): a directory, since mkdir is atomic. Its owner file holds the
# holder's PID; a lock whose holder is gone is taken over.
gpu_lock() {   # <lock dir> <what for>
  local lock=$1 waited=
  mkdir -p "$(dirname "$lock")"
  until mkdir "$lock" 2>/dev/null; do
    local pid
    pid=$(awk '{print $1}' "$lock/owner" 2>/dev/null)
    [ "$pid" = "$$" ] && return 0   # ours already (a retry)
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      rm -rf "$lock"   # stale
      continue
    fi
    [ -z "$waited" ] && waited=1 && echo "$(date +%T) waiting for the GPU lock ($(cat "$lock/owner" 2>/dev/null)) for: $2" >&2
    sleep 5
  done
  echo "$$ $(date +%T) $2" > "$lock/owner"
}
gpu_unlock() {   # <lock dir>: only the holder's own lock
  local pid
  pid=$(awk '{print $1}' "$1/owner" 2>/dev/null)
  [ "$pid" = "$$" ] && rm -rf "$1"
}
