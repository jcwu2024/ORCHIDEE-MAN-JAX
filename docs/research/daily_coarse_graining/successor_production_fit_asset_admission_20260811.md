# Successor Production-fit Asset Admission

Status: **accepted free CPU gate; replacement paid fit awaits explicit
approval**.

Date: 2026-08-11.

## Result

The exact production-fit asset admission passed on `test01` at commit
`0ed98918aa0ae29727282b2f14ab2c67737b95c6`. The probe used the same immutable
500-entry prepared product, runner contract, checkpoint handoff, frozen
coefficient contract and coefficient source report required by the paid fit.
It executed zero optimizer updates and wrote no checkpoint.

The report is:

```text
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/training/
successor-production-admission-0ed9891/admission_report.json
```

Report SHA256:
`50d090e6382e13d3f735fd2d9a92c2feb0fbc9f0c6fa2acb55ecd6d2418ea089`.

## Admitted Identities

- prepared manifest SHA256:
  `0182b7f7a6d5eae09025a336f26bc67ba5fb54706c3529a846d3885c16d80462`;
- producer Git HEAD: `d4936996635b02fcf04cd5f8c4926be3cc010124`;
- consumer Git HEAD: `0ed98918aa0ae29727282b2f14ab2c67737b95c6`;
- formal 500-update schedule SHA256:
  `f6c0d36f500f1dc47699cd8783c0bbf2379a9d0918751d9f1b99280e6a74103b`;
- coefficient source schedule SHA256:
  `2c6ff41808b97b833850ecee667ddeb2e0561ad9d64dba7ce18835a7ce49fadc`;
- byte-identical first artifact SHA256:
  `d0b0fdaee6275546f5d740c670846a6d8e0849b0fa0c1d25ea36afe2987349ee`;
- checkpoint handoff contract SHA256:
  `a587b8a93b57bb69f06776aa80fd2268f6ebe2f2e13c07b70e45a3d6632315b0`;
- frozen coefficient contract SHA256:
  `b32f6cd0102d515b6f556e5e56b170229b27d913af8c0c2172b1e44c1f5f07ad`;
- coefficient source report SHA256:
  `481581d1648699e140fa12c1b6a8e342f154ecb2216473f01a6916bd95c31f60`.

All four resolved coefficient values match the accepted source report. The
probe reports `teacher_data_regenerated=false`, `historical_v5_used=false`,
`sealed_test_used=false`, `training_updates_executed=0`, and
`checkpoint_written=false`.

The Python admission itself took `79.82` seconds. The complete wrapper used
`2:14.73` wall time and peaked at approximately `1.46 GiB` RSS on the free CPU
test node.

## Decision

The source-versus-consuming schedule ownership defect that stopped job
`14517771` is closed against the exact production assets. This gate authorizes
preparation of a fresh paid-resource approval block; it does not submit the
job and does not establish any training or scientific result.
