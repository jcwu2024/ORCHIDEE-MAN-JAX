# Target-Oriented Update-1000 Preparation Acceptance (2026-08-25)

## Status

The complete prepared continuation range for absolute updates 251 through
1000 is accepted. All four free-CPU workers exited zero, and the merge on
`test01` independently verified every artifact hash and deserialized payload.
Paid update-1000 continuation may use this manifest with the exact admitted
update-250 stage checkpoint.

## Prepared Product

- root:
  `runtime/outputs/training/target-oriented-prepared-1000-a243029`;
- merged manifest SHA256:
  `8205140ddeae34967a287d41d204318659a84a8574de954e9b0be14f8998fc31`;
- source commit:
  `a243029e6043d71820c86eeb52a9a42215f9fa91`;
- coverage: exactly 750 unique, contiguous artifacts for updates `251..1000`;
- total artifact bytes: `2,523,051,773`;
- full schedule SHA256:
  `ee1ff380a18e1ff3ea919c579455db5895fa95116f2f95d144d21a852da9e2a5`;
- training contract SHA256:
  `91faf01123008a60c59038c074779a2a2b4553b82c3739475dd4829fe91ea25b`;
- dataset manifest SHA256:
  `5fc2c0691bfa53252f875f7646f3c1a5bea203c2759a6e947b26bb51dc13aa99`;
- retained executable trace signature:
  `13ed0a2eb5895d9b9c8325a1e122a8e07d51a615fb842676b1ac17cf0cc51d8b`.

The four admitted chunk manifest hashes are:

| Absolute range | Artifacts | Manifest SHA256 |
| --- | ---: | --- |
| `251..438` | 188 | `726d9521ca1967624392fda998f23a2bec6a3f4c7b90f21c418bd1355abb7dbe` |
| `439..626` | 188 | `d24919cdd8dca1087c8989e1d5e66043033badcf74e013ead868b136313fd4ef` |
| `627..813` | 187 | `84f6f7bc5fdbe3a8a89b02f16a402df19fc1b9c1f18f09295c26ac51b0023d8d` |
| `814..1000` | 187 | `bc5bbf7a3e6444f3927a978403c09f2a75667e5bddeab3cf05f5138affc48e6e` |

The merged manifest declares `teacher_data_regenerated=false`,
`historical_v5_used=false`, and `sealed_test_read=false`. Preparation reused
only retained-input scaffolding from the accepted update-1-to-250 product; it
did not copy those updates into this continuation range.

## Next Operation

Run the dedicated paid launcher
`scripts/hpc/slurm_target_oriented_training_1000.sh`. It must restore
`stage-update-0250.pkl` with SHA256
`099d48c76277cafb04daf868c492200d0271fd6a4a2c6fa29ca6e4bdbe7989a5`,
including AdamW step and moments, and retain exactly updates 251 through 1000.
The frozen 96-window validation panel controls the update-1000 decision.
