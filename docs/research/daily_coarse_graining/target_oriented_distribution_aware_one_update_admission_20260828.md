# Distribution-Aware One-Update Admission

Date: 2026-08-28

Status: accepted free-GPU optimizer admission; no update retained

Executable commit:
`5ea0033177e7989616480ef140b7a994a7ce0472`

Bounded training contract SHA256:
`0c60924ee12d0ea555b082f76990ca55e34f86f67effb052f5eec44c01210051`

Report SHA256:
`a32b2c175b153d8ac7c732b70a1bf9b2124fbe49a122ab7414f16a7842c7734e`

## Result

The artifact-252 ephemeral update passed on free `gln01`. The free, Teacher-
guard, and combined gradients each covered all `9,750,529` adapter parameters;
all 36 leaves were finite and nonzero. Their norms were `0.6653940`,
`21.2146895`, and `2.0161371`, respectively.

The combined norm remained below the frozen `2.6926205` threshold, so the
clip scale was exactly `1.0`. Fresh AdamW update 1 used learning rate `4e-6`.
All adapter leaves changed, with total step norm `0.0114151`, while all
`83,754,404` retained parent parameters remained exactly unchanged by object
identity.

The same-batch combined objective decreased from `1.1132878799` to
`1.1130876797`. The free objective remained essentially level while the guard
decreased; no conclusion about generalization is drawn from this single
update. Free and Teacher-state hard-failure counts were zero before and after.

Peak live JAX allocation was `2,659,083,520` bytes. End-to-end elapsed time
was about 17.8 minutes, including compilation; the actual measured update
path took about 15.7 minutes. No update, optimizer, checkpoint, or recovery was
retained, and no validation or sealed-test value was read.

## Decision

The adapter-only optimizer path is admitted. The next gate is local
qualification and separate paid-resource approval of the exact 250-update
training runner. The Attention parent remains the selected backbone.

