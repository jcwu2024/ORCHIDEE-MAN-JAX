# Coupled GPP Direct-Step Transfer Result (2026-09-04)

## Outcome

The bounded free-`gln01` audit completed at commit `48eb1ae` in
`3,935.020` seconds. Its classification is
`train_only_scale_fails_monitor`: the train-only selector chose scale `1/4`,
but direct-GPP mean loss on the fixed monitor increased by `0.5107%`.

The server report is
`runtime/outputs/diagnostics/target-oriented-coupled-gpp-direct-step-transfer-48eb1ae.json`,
SHA256
`df2593b423c3f380f6af45b7ee3faaeddae0f78a29911371928b54a91a966c94`.
The hash identifies the immutable report; it is not a cross-GPU numerical
equality gate.

## Train-Only Scale Selection

The same persistent-update-2 direction was evaluated at all six frozen
scales. Negative percentages are improvements. The scheduled columns are
direct GPP, reachable one-day state, 7-day science, and 30-day science. The
four guard columns are disjoint train/train direct-GPP groups that were not
used in persistent updates 1 through 10.

| Scale | Scheduled GPP | State | 7 day | 30 day | Guard 1 | Guard 2 | Guard 3 | Guard 4 | Combined guard | Pass |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `1/2` | `-4.9432%` | `-4.0348%` | `-3.8495%` | `-2.9431%` | `+0.3213%` | `-1.2141%` | `-1.6151%` | `-0.2996%` | `-0.6450%` | no |
| `1/4` | `-3.6708%` | `-2.2814%` | `-1.9606%` | `-1.0764%` | `-0.2811%` | `-1.2529%` | `-1.7612%` | `-1.0653%` | `-1.0303%` | yes |
| `1/8` | `-2.1160%` | `-1.1765%` | `-0.9736%` | `-0.3483%` | `-0.2264%` | `-0.7641%` | `-1.1062%` | `-0.7223%` | `-0.6670%` | yes |
| `1/16` | `-1.1257%` | `-0.5923%` | `-0.3268%` | `-0.0262%` | `-0.1317%` | `-0.4123%` | `-0.6072%` | `-0.4027%` | `-0.3677%` | yes |
| `1/32` | `-0.5795%` | `-0.2973%` | `-0.1367%` | `+0.1428%` | `-0.0702%` | `-0.2131%` | `-0.3168%` | `-0.2110%` | `-0.1919%` | no |
| `1/64` | `-0.2939%` | `-0.1488%` | `-0.0626%` | `+0.2543%` | `-0.0361%` | `-0.1082%` | `-0.1617%` | `-0.1078%` | `-0.0979%` | no |

Scale `1/2` fails because guard group 1 regresses. Scales `1/32` and `1/64`
fail because the scheduled 30-day objective regresses. The largest shared
train-only passing scale is therefore `1/4`; the monitor was not read until
that selection was complete.

## Fixed-Monitor Transfer

At scale `1/4`, fixed-monitor direct-GPP mean loss changes from
`0.0629695005` to `0.0632910878`, or `+0.510703%`. Four of eight artifacts
improve and four regress. Individual relative changes are:

| Artifact update | Relative change |
| ---: | ---: |
| `251` | `-3.7510%` |
| `358` | `+9.2311%` |
| `465` | `+2.1506%` |
| `572` | `-10.2742%` |
| `679` | `-3.3303%` |
| `786` | `+13.1346%` |
| `893` | `-4.0236%` |
| `1000` | `+1.6396%` |

The smaller step reduces the earlier scale-`1/2` monitor regression from
`2.2197%` to `0.5107%`, so finite-step size is part of the failure. It does
not change the sign on the independent monitor, and the preregistered
transfer gate therefore fails.

## Integrity And Runtime

All scheduled physical gates pass at every scale. The parent recovery,
direct-generalization report, and all `93,504,164` fixed parameters remain
unchanged. No optimizer update, parameter proposal, recovery, checkpoint, or
trainable parameter is written. The audit uses only accepted v6 train/train
artifacts and does not read fresh validation or sealed test data, regenerate
Teacher data, join v5 data, reconstruct 48 forcing records, or execute a
48-step state scan.

Reported JAX allocator peak is `4,098,246,656` bytes, so this bounded audit is
not memory limited on the single V100.

## Decision

Close the single-group direct-step transfer question. Do not continue the
retained recovery, inspect more monitor scales, or change the Attention
architecture from this result.

The next task is a frozen zero-retention direction-construction audit at the
same pre-update-2 state. Direct GPP from the scheduled group and the four
train-only guard groups must enter as separate direction constraints alongside
reachable state, 7-day science, and 30-day science. Scale selection remains
train-only, and the fixed monitor may receive only the one selected proposal.
This tests whether the update rule can construct a genuinely shared descent
direction before any further persistent training is considered.
