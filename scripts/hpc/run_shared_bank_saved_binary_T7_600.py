"""Pending fresh600s free-GPU same-binary acceptance; no automatic execution."""
import run_shared_bank_saved_binary_T7 as runner

runner.TOTAL_CAP=600
runner.MAIN_CAP=590
runner.previous.GLOBAL_CAP=600
runner.FAMILY=runner.ROOT/'runtime/outputs/diagnostics/shared-bank-adjoint-saved-binary-T7-600-20261003'

if __name__=='__main__':raise SystemExit(runner.main())
