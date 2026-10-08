# Root-owned immutable deployment recipe

Do not execute before root supplies the accepted commit SHA. No submission is part of this recipe. Endpoint metadata confirms Git1.8.3.1 and immutable v3 standalone `.git`, HEAD12b495b9425de3ad1dc16efa3a5a0e69d7e89222. Use only an independent `git clone --no-hardlinks --no-checkout` from immutable v3, then fetch the incremental bundle into the NEW repository and check out the exact SHA. The script uses subshell `cd` and sequential Git commands; old working files, refs and objects are not modified. Root executes on test02; login is limited to read-only/scheduling/SCP transport.

After root commit, create a bundle in the local worktree:

```powershell
git bundle create outputs/research/global_throughput/gpu-metadata-repair-20261008/global-throughput-gpu-metadata-v1.bundle HEAD '^12b495b9425de3ad1dc16efa3a5a0e69d7e89222'
```

Transfer that bundle and `deploy_new_checkout.sh` through the working `.2` login endpoint to a new path under `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/`. Preserve normal host-key checks and the existing SSH identity. `git bundle verify` requires the previously deployed v3 base objects; if the new clone lacks them, stop and report the missing prerequisite to root. Run on test02 as bounded filesystem/Git transfer work, with root's full accepted SHA and immutable v3 repository path:

```bash
bash deploy_new_checkout.sh ACCEPTED_FULL_SHA /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfers/global-throughput-gpu-metadata-v1.bundle /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-sweetspot-v3
```

The third argument is the inspected immutable v3 standalone repository, used read-only. The script admits only that path and HEAD. All bundle verification/fetch/checkout and clean-status checks run inside the new independent metadata-v1 repository. No existing branch is advanced; a later existing-branch update would require fast-forward-only handling.

On `test02`, one outer `timeout60` canonical CPU interface window should set `CODE_DIR` to the new immutable checkout, `cd` there, use the canonical `orcjax_cpu/bin/python`, and perform: `source_identity(Path.cwd())` asserting active certificate/canonical science11b6…; actual `benchmark_global_gpu_capacity.py --help`; actual CLI `--dry-run --continue-512 --budget-seconds2700` with all declared proposal source flags; the new512 wrapper `--dry-run`; and a copied Slurm-spool wrapper `--dry-run` retaining `CODE_DIR`. Flag/value pairs are separate in actual commands. Save stdout, exact code SHA, source identity, and elapsed outer wall in root's deployment receipt. No JAX import, model preparation, lowering, compilation, GPU initialization or invocation is needed. The unchanged CUDA `global_batch_gpu_entry.py` remains runtime-pending.

Before this final window, canonical diagnostic wall is353.5285148s; remaining cap is246.4714852s. A successful≤60s interface window leaves total≤413.5285148s, well within600s. This does not add a new model-probe allowance. Paid approval is still separate and pending after the concrete deployed receipt.
