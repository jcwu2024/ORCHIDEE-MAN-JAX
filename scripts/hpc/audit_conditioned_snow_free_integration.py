"""Run one corrected-head train-only proposal, retaining no weights."""
from scripts.hpc.audit_conditioned_attention_streamed_integration import _parser, run

if __name__ == "__main__":
    run(_parser().parse_args(), warm_snow_free_boundary=True)
