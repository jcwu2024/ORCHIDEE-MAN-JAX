# Structured GPP Joint-Gradient Heterogeneity Result

Date: 2026-08-29

Status: completed and accepted; effective batch size 4 selected for a
discarded accumulated-AdamW preflight

Execution commit:
`95322f8eb6fca4ff5cf9ac4d722cec11710cd1a5`

Audit contract SHA256:
`0905b11467e982cdff72c624f2c61c2849e4e2dd90cbee860b9996c9588d945b`

Report SHA256:
`15d993a2c184918c7b6feb246a246d4d0ee8ecabfcdefb240abfd0659c880eea`

## Result

The free-`gln01` zero-update audit completed on the eight frozen train/train
artifacts. All 24 source gradients and all grouped gradients are finite,
nonzero, cover all `12,879,363` specialist parameters, and reach all 129
specialist leaves. All one-day, 7-day, and 30-day physical hard-failure counts
are zero. The `93,504,164`-parameter parent and the warm-start checkpoint are
unchanged.

The smallest effective batch size for which every nonoverlapping group has a
strictly positive direct-GPP, 7-day, and 30-day cosine with the calibrated
combined gradient is 4:

| Effective batch size | Passing groups | Total groups | Classification |
| ---: | ---: | ---: | --- |
| 1 | 7 | 8 | inconsistent |
| 2 | 3 | 4 | inconsistent |
| 4 | 2 | 2 | consistent |
| 8 | 1 | 1 | consistent but not minimum |

The two admitted batch-four groups are:

| Artifact updates | Direct cosine | 7-day cosine | 30-day cosine |
| --- | ---: | ---: | ---: |
| `512, 548, 584, 620` | `0.9559` | `0.8639` | `0.7992` |
| `656, 692, 728, 763` | `0.9584` | `0.8481` | `0.7543` |

The first group includes artifact 512, whose single-artifact proposal had a
negative 30-day direction. Its positive batch-four result therefore directly
shows that the earlier failure was a sampling-unit problem rather than proof
that the three objectives or architecture are irreconcilable.

## Execution

The audit took `977.16 s`. Peak live JAX allocation was `2,647,203,072` bytes,
approximately `2.47 GiB`. It created no optimizer, executed no update, and
wrote no checkpoint. Teacher data was not regenerated; historical v5,
validation, and sealed-test values were not read; no 48-record forcing
reconstruction or subdaily scan was used.

## Decision

Batch size 4 is frozen as the minimum consistent effective batch for the next
diagnostic. This result does not authorize paid training. The next gate is two
independent discarded AdamW proposals, one for each batch-four group, both
starting from the same warm-start specialist and optimizer state at step 250.
The gate applies to the three group-mean losses; individual artifact changes
are reported but are not separately thresholded.

