"""Pending new500s free-GPU full T7 values/pairs using reviewed actual IR proof."""
import run_shared_bank_saved_binary_T7 as runner

runner.TOTAL_CAP=500
runner.MAIN_CAP=490
runner.REQUIRE_SAVED_PROOF=True
runner.previous.GLOBAL_CAP=500
runner.FAMILY=runner.ROOT/'runtime/outputs/diagnostics/shared-bank-adjoint-proved-binary-T7-20261003'

if __name__=='__main__':raise SystemExit(runner.main())
