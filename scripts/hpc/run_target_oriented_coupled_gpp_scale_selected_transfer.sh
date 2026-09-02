#!/bin/bash
# Run the free-gln01, no-checkpoint 7/30-day transfer audit.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export RUN_KIND=transfer
exec "$SCRIPT_DIR/run_target_oriented_coupled_gpp_scale_selecting_recalibration.sh"
