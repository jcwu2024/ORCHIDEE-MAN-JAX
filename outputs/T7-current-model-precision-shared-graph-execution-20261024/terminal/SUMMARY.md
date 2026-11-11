# Terminal recovery — job16060441

Recovered read-only on2026-11-11; no rerun. Authoritative compact fields are in
[summary.json](summary.json), scheduler receipt in [sacct-terminal.txt](sacct-terminal.txt),
and original logs/result/supervisor/ledger beside them. All copied remote files
match [remote-sha256.txt](remote-sha256.txt).

- Source `96978caf04c9a56273ea7e9b9031f57aebf999ac`; node ibc13b03n03,4CPU/1V100.
- Actual allocation2026-10-26 11:20:43–12:17:52 Asia/Shanghai,3429s;
  extern step3430s. Job/batch FAILED,124:0.
- `incomplete_timeout`, `deadline_exceeded`, phase **compile**. One trace/lower
  completed; one compile attempted, zero completed; zero model-call attempts.
  All52 planned numerical checks unstarted; no native/FD arrays or verdict.
- Lower-start37.640663s; compile-start466.218226s; deadline3337.641255s.
  The428.577563s interval includes lower/artifact/inspection, not pure lower time.
  Lower+compile shared3300s graph budget was enforced. Preflight passed68.620025s.
- Saved `result.v2.lowered.mlir.gz`, SHA256
  `552ffd1aeb52385261801dbc3e592b0e853db5789936a68097c0a170ead2a1fc`.
- Retained inputs consumed, no missing static fields, original prepared case unchanged.
  State20 rawNaN remains an independent strict failure; no new trajectory audit.
- GPU/process-group release verified:0MiB/0%, no GPU processes or owned groups.
  Sampled peak1525MiB does not certify complete-model memory admission.
- Supervisor occupation3428.596681s, estimated¥2.361922; sacct3429s estimate¥2.3622
  at4×0.07+2.20 CNY/hour. Actual bill unknown; approved maximum¥3.72.

The one-shot approval is consumed. No retry, longer graph, watcher or training
is authorized by this recovery. This is a compile-time resource outcome, not a
numerical acceptance or rejection of the52 unexecuted checks.
