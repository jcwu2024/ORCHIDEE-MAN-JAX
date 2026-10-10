"""Preserve original map/contractions and screen native execution scheduling."""
import sys
from probe_shared_chunk_group_component import main

if __name__ == "__main__":
    sys.argv.extend(("--execution-variant", "latency-schedule"))
    main()
