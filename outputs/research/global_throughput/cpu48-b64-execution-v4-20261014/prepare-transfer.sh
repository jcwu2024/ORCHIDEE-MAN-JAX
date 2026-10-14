set -euo pipefail
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
T=$P/runtime/transfer/global-cpu48-b64-v4-60dccbddde10
case "$(readlink -f "$P")" in /WORK/liwei_work/jcwu/*) ;; *) echo "project path escapes authorized root" >&2; exit 2;; esac
test ! -e "$T"
test ! -e "$P/runtime/worktrees/global-throughput-cpu48-b64-v4"
mkdir "$T"
mkdir -p "$P/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v4-20261014/scalar-audit"
mkdir -p "$P/runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014"
mkdir -p "$P/runtime/logs"
date -Is
readlink -f "$T"
