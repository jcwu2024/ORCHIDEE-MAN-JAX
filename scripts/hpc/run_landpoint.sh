#!/usr/bin/env bash
set -euo pipefail

: "${ORCHIDEE_DATA_ROOT:?set ORCHIDEE_DATA_ROOT}"
: "${ORCHIDEE_REFERENCE_ROOT:?set ORCHIDEE_REFERENCE_ROOT}"
: "${ORCHIDEE_OUTPUT_ROOT:?set ORCHIDEE_OUTPUT_ROOT}"
: "${LANDPOINT_ID:?set LANDPOINT_ID}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
point_output="${ORCHIDEE_OUTPUT_ROOT}/acceptance/${LANDPOINT_ID}"
mkdir -p "${point_output}"

cd "${repo_root}"
exec uv run --frozen orchidee-jax validate-landpoints \
  --landpoint-id "${LANDPOINT_ID}" \
  --start-year "${START_YEAR:-1961}" \
  --years "${YEARS:-50}" \
  --initial-state cold-start \
  --strict-on-failure off \
  --output-dir "${point_output}"
