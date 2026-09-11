# DOC-only input tail: zero-update mechanism separation

Frozen before execution. This is a narrower candidate and causal test following
the completed all-state T=16 screen. It is not automatic promotion of that run.

All-state compression fixed the numerical mismatch on all three cases, but
changed other state features and regressed the 883/30-day science total from
0.074534 to 0.254498. The 322/7-day total improved from 0.285553 to 0.001098,
and 442/30-day improved from 0.179439 to 0.144905. Maintenance worsened on all
three. Thus reject the all-state transform as a drop-in update-41 correction;
this does not show it cannot be trained, or that Attention is unsuitable.

The completed isolated coordinate probe now proves DOC coordinate 1785 alone
causes the same 0.2896344 day-10 GPP jump as replacing the whole DOC leaf.
All-state compression removes that coordinate response (exactly zero), and
reduces the whole-leaf response to 7.96e-7 gC m-2 day-1. Other fields' causal
contributions to the changed science errors remain unknown.

## Candidate and matched reference

Apply the existing C1 log tail with unchanged T=16 only to coordinates owned
by `slowproc_stomate_previous_step_state.DOC`, resolved from the canonical
state contract (current slice 1327:2223). This is a named process field rule,
not a hard-coded special treatment of one numerical coordinate. All other
continuous features use exactly their legacy arithmetic, including precision.
The mask propagates through summary, PFT, coordinate-local, structured-GPP and
carbon-feedback input routes. Raw physical state remains unchanged. The mask
and threshold default to None, leaving the ordinary trained model unchanged.

Weights: accepted update 41, no updates. Retain Attention topology, existing
trained/frozen modules, data v6, statistics, losses and original tolerances.
Train cases remain exactly (883,30,3), (322,7,1), (442,30,0). Do not tune T on
these results or access validation. The unchanged reference and all-state arm
are already completely measured at fe3f6fd in
`conditioned-input-tail-14991510.json`; reuse those reference rows instead of
recompiling the same unchanged model. Report the source path explicitly.

## Bounded gate

Local tests first: default compatibility; bitwise unchanged unselected features
and their derivatives; selected finite C1 transform; mask indexing in field
routes; forward/AD with extreme selected values; existing gate classifier.

Then free gln01, one idle V100, 45-minute cap, no optimizer or parameter writes.
Run the same saved day-10 leaf and coordinate interventions, all three cases'
production and compact traces, and actual physical-unit curves. Recompute all
panel sufficient statistics for the candidate. Require score atol=1e-6,
rtol=1e-5, paired trajectory masks, finite outputs and zero physical hard
failures. Require day-10 probe differences <=1e-5 gC m-2 day-1. Keep the prior
5%+1e-8 per-case/per-science-MSE regression screen unchanged; do not relax it
after seeing the result. Show physical cumulative GPP/NPP/respiration and
AGB/BGB net changes even if that gate fails.

If these forward gates pass, proceed to the existing real-model derivative
admission with all helpers rebuilt for the masked normalizer. If not, stop
drop-in promotion and interpret the two fixed-scope arms together: they
separate numerical stabilization from inherited-model dependence on the old
input distribution. No paid training or gradient acceptance follows merely
from score equality. Any subsequent adaptation training needs an explicit
method, budget and approval, with reproducible final-training provenance.
