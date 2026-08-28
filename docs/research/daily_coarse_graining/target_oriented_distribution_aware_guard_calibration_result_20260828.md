# Distribution-Aware Teacher-Guard Calibration Result

Date: 2026-08-28

Status: accepted zero-update calibration; coefficient frozen for the bounded
adapter-only training contract

Executable commit:
`b980cdee3f7d24dbb12f5643800150db6df293ba`

Calibration contract SHA256:
`c101940963465df1e25878f596645037e7d2de22dcefcf21f622fcd1de7179d7`

Report SHA256:
`bb55388f598cc6e4e421597be3b988e7d5d12ab1d66a5e5528b2883addf7a474`

## Result

The frozen eight-batch train/train calibration passed on free `gln01`. Every
free-objective and Teacher-guard gradient covered all `9,750,529` adapter
parameters, all 36 leaves were nonzero, every value was finite, and all
physical hard-failure counts were zero.

The free-objective gradient norms were:

```text
0.3073625040, 0.7823603694, 0.2596275356, 0.3895532722,
0.1488654966, 0.3288700727, 0.5233826519, 0.1853282859
```

The Teacher-day-start guard gradient norms were:

```text
21.4245691138, 18.4241710609, 7.0046768773, 8.7844303921,
20.3529288269, 25.2205071441, 9.4823135001, 8.2824791679
```

Their medians are `0.31811628835001987` and `13.953242280504194`.
The raw ratio is `0.022798736089783205`; the preregistered `[0.1, 10.0]`
bound therefore freezes the Teacher-guard coefficient at exactly `0.1`.
This value cannot be retuned from validation or rollout results.

The first batch paid the expected compilation cost. The remaining seven hot
batches each took approximately 44 seconds. Peak live JAX allocation was
`2,690,634,496` bytes. The run created no optimizer, update, checkpoint, or
recovery and read no validation or sealed-test values.

## Decision

Calibration is accepted. The next gate is a separate adapter-only bounded
training contract that binds the numeric coefficient and this report hash.
The update-1000 coordinate-aligned Attention parent remains frozen; this
result does not support replacing the Attention backbone.

