from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.daily_coarse_graining.physical_parameter_registry import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    load_physical_parameter_registry,
    registry_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the physical-parameter candidate registry")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    args = parser.parse_args()
    registry = load_physical_parameter_registry(args.registry, root=ROOT)
    print(json.dumps(registry_summary(registry), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
