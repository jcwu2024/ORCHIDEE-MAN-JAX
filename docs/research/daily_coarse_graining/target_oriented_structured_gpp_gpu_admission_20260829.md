# Target-Oriented Structured GPP Real-V6 GPU Admission

Date: 2026-08-29

Status: accepted zero-update real-v6 GPU admission

Commit: `dbc137346ccfc98d2918502374903ba4b1c3a7a2`

Architecture contract SHA256:
`ce211fa7c6f5a09bd900e02e31a864b61fb32f6b06ee96d4262402c513d620ea`

Admission report SHA256:
`7a5d4310912494b8906d870d95fa04de44d05b471e9aff4350f9cbd3ec04c488`

Remote report:
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/target-oriented-structured-gpp-real-v6-dbc1373.json`

Local report mirror:
`outputs/research/daily_coarse_graining/target-oriented-structured-gpp-real-v6-dbc1373.json`

## Execution

The admission ran directly on free `gln01`, physical host `ibc13b02n04`, on
one Tesla V100-SXM2-32GB. GPU 1 was selected only after `nvidia-smi` showed it
idle. No Slurm job or paid allocation was used.

The runner consumed the accepted v6 train/train day for landpoint
`001.0-071.0`, year 1961, day index 2. It loaded the update-1000 Attention
checkpoint and update-250 distribution-aware maintenance adapter, then
initialized only the new structured GPP specialist. Validation artifacts were
read only for frozen checkpoint identity; no validation values were
evaluated.

## Result

The report status is `passed` and the backend is `gpu`.

- parent parameter count: `93,504,933`;
- removed old GPP projection: `769`;
- structured specialist count: `12,879,363`;
- complete candidate count: `106,383,527`;
- all parent array objects and all non-GPP primary heads were preserved;
- all 160 non-GPP learned outputs equal the parent bit for bit;
- eager and compiled complete outputs are finite;
- eager/JIT comparison passes with maximum tolerance ratio `0.59545`;
- JIT replay time is `0.0640 s` for the production-shaped single-day forward;
- complete admission proxy loss is finite at `1149.175797887741`;
- all `12,879,363` specialist-gradient elements are finite;
- all `129/129` specialist gradient leaves are nonzero;
- forcing encoder, parent/PFT context, potential, limitation, correction, and
  all three output projections each have finite nonzero gradient norm;
- all 181 canonical named outputs are present after the retained tail;
- all 5,737 declared next-state elements are finite;
- water and carbon budget residuals are both exactly zero.

The measured live device allocation was `487,971,328` bytes before candidate
materialization, `625,421,312` bytes after all gates, and
`1,471,938,560` bytes at peak. The approximately 25 GiB visible in
`nvidia-smi` was allocator reservation, not peak live tensor use.

Compilation and execution timings were `37.79 s` eager, `191.03 s` for the
first compiled forward, `170.17 s` for the full specialist gradient, and
`240.49 s` for the canonical transition compile and run. XLA emitted one
nonfatal persistent-cache parse warning and then recompiled. The later JIT
replay and every numerical gate passed, so the warning is classified as a
cache miss rather than a model or runner failure.

## Policy Closure

The report records:

- training updates: `0`;
- optimizer created: `false`;
- checkpoint written: `false`;
- parent parameters trainable: `false`;
- Teacher data regenerated: `false`;
- historical v5 used: `false`;
- validation values evaluated: `false`;
- sealed test read: `false`;
- forcing records reconstructed: `0`;
- subdaily scan steps: `0`.

The local and real-v6 implementation gates are now complete. The next gate is
to freeze a separate bounded training contract before any optimizer update.
That contract must define the train/train data schedule, specialist-only
parameter ownership, one-day direct GPP anchor, 7/30-day long-horizon loss,
clipping/learning-rate calibration, recovery, early stopping, resources, and
absolute promotion gates. It must not authorize validation, 365-day rollout,
Gate D2, sealed test, or product acceptance.
