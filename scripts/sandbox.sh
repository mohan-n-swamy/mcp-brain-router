#!/usr/bin/env bash
# Review and tool-worker sandbox (lean-rig PLAN §6; proven by policy-check.sh, I22).
#
# Runs one CLI under bubblewrap. Inside it sees its worktree (read-only for a
# review), the login dirs named with --login, the read-only paths named with
# --ro, and the system dirs (/usr, /etc). The rest of $HOME and /tmp are empty
# tmpfs, and the env holds only an allowlist. Results are untrusted worker
# output: the caller reads them as data.
#
# Native mechanism it could not use: each CLI's own sandbox differs (Grok has
# none, Codex's needs its own bwrap), so one bwrap wrapper serves them all.
#
# Usage: sandbox.sh [--review] [--class personal|orange] [--login DIR]...
#                   [--ro PATH]... [--env NAME]... WORKTREE -- CMD [ARG]...
#   --review      mount WORKTREE read-only (every review courier uses this)
#   --class C     data class. orange is refused: no review or tool worker runs
#                 on Orange work. Default: default_class in the router config
#                 (orange on the Orange host), else personal.
#   --login DIR   that CLI's login dir under $HOME, read-write (token refresh)
#   --ro PATH     an extra read-only path: the CLI's install dir, a prompt file
#   --env NAME    pass one more variable through; secrets are never default
#
# Example (a Grok review, one tools-off turn, diff in a prompt file):
#   sandbox.sh --review --login ~/.grok --ro ~/.grok/bin --ro /tmp/review.txt \
#     ~/work/repo -- ~/.grok/bin/grok --prompt-file /tmp/review.txt \
#     --max-turns 1 --no-subagents --no-plan --disable-web-search
#
# bwrap: /usr/bin/bwrap, root-owned (gate 2). BRAIN_ROUTER_BWRAP overrides it,
# for a test run before gate 2 lands.
# Exit: the command's status; 2 infra (not Linux, no bwrap); 3 refused; 64 usage.
set -euo pipefail

usage() { sed -n '14,22p' "$0" >&2; exit 64; }
die() { echo "sandbox: $2" >&2; exit "$1"; }

[[ $(uname -s) == Linux ]] || die 2 "Linux only (bwrap)"
BWRAP=${BRAIN_ROUTER_BWRAP:-/usr/bin/bwrap}
[[ -x $BWRAP ]] || die 2 "no bwrap at $BWRAP (gate 2 installs /usr/bin/bwrap)"
[[ -n ${HOME:-} && -d $HOME ]] || die 2 "HOME is not a directory"
home=$(realpath "$HOME")

review=0 class="" logins=() ros=() envs=()
while (($#)); do
  case $1 in
    --review) review=1; shift ;;
    --class) [[ $# -ge 2 ]] || usage; class=$2; shift 2 ;;
    --login) [[ $# -ge 2 ]] || usage; logins+=("$2"); shift 2 ;;
    --ro) [[ $# -ge 2 ]] || usage; ros+=("$2"); shift 2 ;;
    --env) [[ $# -ge 2 ]] || usage; envs+=("$2"); shift 2 ;;
    -h|--help) usage ;;
    --) usage ;;
    -*) die 64 "unknown option $1" ;;
    *) break ;;
  esac
done
[[ $# -ge 3 && $2 == -- ]] || usage
wt_arg=$1; shift 2

# Class: the caller's, else the host default from the router config (one key
# read; nothing else in that file is printed), else personal.
if [[ -z $class ]]; then
  cfg=$home/.config/mcp-brain-router/config.toml
  if [[ -r $cfg ]]; then
    class=$(sed -n 's/^[[:space:]]*default_class[[:space:]]*=[[:space:]]*"\([A-Za-z]*\)".*/\1/p' "$cfg" | head -n 1)
  fi
  class=${class:-personal}
fi
case ${class,,} in
  personal) ;;
  orange) die 3 "class orange: no review or tool worker runs on Orange work" ;;
  *) die 64 "unknown class '$class' (personal or orange)" ;;
esac

# A path the sandbox shows must exist and must not be $HOME, /, or a parent of
# $HOME: any of those would show the whole home, keys included.
shown() {
  local p
  p=$(realpath -e -- "$1" 2>/dev/null) || die 64 "no such path: $1"
  case "$home/" in "$p"/*) die 3 "refusing to show $p: it holds \$HOME" ;; esac
  [[ $p != / ]] || die 3 "refusing to show /"
  printf '%s' "$p"
}
wt=$(shown "$wt_arg")
[[ -d $wt ]] || die 64 "worktree is not a directory: $wt_arg"

args=(--unshare-all --share-net --die-with-parent --new-session --cap-drop ALL
      --ro-bind /usr /usr --ro-bind /etc /etc
      --ro-bind-try /run/systemd/resolve /run/systemd/resolve
      --proc /proc --dev /dev --tmpfs /tmp --tmpfs /var/tmp --tmpfs "$home")
for d in /bin /sbin /lib /lib32 /lib64; do
  if [[ -L $d ]]; then args+=(--symlink "$(readlink "$d")" "$d")
  elif [[ -d $d ]]; then args+=(--ro-bind "$d" "$d"); fi
done
for p in "${ros[@]}"; do p=$(shown "$p"); args+=(--ro-bind "$p" "$p"); done
for p in "${logins[@]}"; do
  p=$(shown "$p")
  [[ $p == "$home"/* ]] || die 64 "a login dir lives under \$HOME: $p"
  args+=(--bind "$p" "$p")
done
if ((review)); then args+=(--ro-bind "$wt" "$wt"); else args+=(--bind "$wt" "$wt"); fi

# Env allowlist: nothing else crosses, so OPENBRAIN_KEY and the like stay out.
args+=(--clearenv --setenv HOME "$home" --setenv PATH "${PATH:-/usr/bin:/bin}")
for name in LANG LC_ALL TERM TZ USER LOGNAME "${envs[@]}"; do
  [[ $name =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || die 64 "bad env name: $name"
  if [[ -n ${!name+set} ]]; then args+=(--setenv "$name" "${!name}"); fi
done

exec "$BWRAP" "${args[@]}" --chdir "$wt" -- "$@"
