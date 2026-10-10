"""Original decoder formula with opt-in native scan unroll=2 scheduling.

Reuses real capture and component gates from probe_shared_normalized_keys;
normalization remains in its original post-gather location for both arms.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

if __name__ == "__main__":
    source = Path(__file__).with_name("probe_shared_normalized_keys.py")
    spec = importlib.util.spec_from_file_location("decoder_chunk_screen_owner", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    sys.argv.extend(("--variant", "unroll2"))
    module.main()
