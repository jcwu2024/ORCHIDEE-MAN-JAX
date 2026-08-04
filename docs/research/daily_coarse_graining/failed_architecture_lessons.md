# Failed Architecture Lessons

Status: active negative-design authority for Gate E and later neural work.

Date: 2026-08-04.

## Purpose

This document records neural hypotheses that were tested and rejected before
the true-daily operator decision. It is a decision index, not a chronological
experiment log. New work should read this page before selecting an
architecture or objective; the dated reports are needed only when checking a
specific result or evidence hash.

"Rejected" here usually means a valid, completed experiment failed its
predeclared scientific gate. It does not mean that the code crashed or that
the experiment was wasted. The negative results constrain the new design.

## Superseded Boundary And Checkpoints

Daily Markov Contract v4 omitted the fast-owned cross-day `leaf_ci` output.
Contract v5 repaired the state boundary losslessly from the stored successor
state. All v4 statistics and neural checkpoints are therefore superseded even
when their original experiments ran correctly. They may be used as historical
evidence only and must not initialize new training.

The first contract-v3 neural pilot is also historical. Its statistics treated
the `rveget=1e20` undefined sentinel as an ordinary regression value and its
fast target duplicated finalize mirrors. Neither its statistics nor its
checkpoint is compatible with the accepted contracts.

## Rejected Design Lessons

| Tested idea | What the evidence showed | Binding lesson for new work | Primary evidence |
| --- | --- | --- | --- |
| Anonymous direct next-state prediction | One-step prediction was learnable, but seven-day teacher-forced errors stayed small while free-rollout error grew sharply. The retained exact daily tail passed independent forward, reverse, and scan checks. | The main failure was recursive state-distribution drift in the learned fast transition, not the retained-tail handoff. Gate E predicts typed process quantities and applies one constrained update instead of predicting an anonymous state vector. | [`autoregressive_training_review_20260725.md`](autoregressive_training_review_20260725.md), [`long_rollout_architecture_review_20260727.md`](long_rollout_architecture_review_20260727.md) |
| Absolute defined-state classification for `rveget` | Predicting absolute defined/undefined state discarded a strong persistence prior. Predicting only status flips reduced classification errors substantially. | Masks, undefined status, and discrete state need explicit transition semantics. They are not ordinary continuous outputs. | [`nine_point_v4_neural_baseline_20260724.md`](nine_point_v4_neural_baseline_20260724.md) |
| Uniform aggregate state loss | The 3,854-value aggregate could look acceptable while biomass, NPP, respiration, `litterpart`, and low-dimensional carbon fields drifted badly. Global scores hid failures in scientifically important small families. | Use process-owned outputs and fieldwise scientific gates. A global RMSE can summarize but cannot promote a model. | [`autoregressive_training_review_20260725.md`](autoregressive_training_review_20260725.md), [`rollout_stability_screening_20260731.md`](rollout_stability_screening_20260731.md) |
| More updates and process/increment loss reweighting | `process_increment_v2` failed every required seen-condition improvement gate, and matched continuation training could worsen all frozen rollout scores. | Do not restart open-ended epoch, coefficient, or loss-weight searches. A new run needs a new mechanism and a bounded A/B hypothesis. | [`process_increment_objective_ab_20260725.md`](process_increment_objective_ab_20260725.md), [`structured_architecture_ab_result_20260726.md`](structured_architecture_ab_result_20260726.md) |
| Stop-gradient on-policy pushforward as the primary fix | It improved seen-point rollout substantially but regressed NPP and growth respiration and worsened spatial and joint validation. | Model-state exposure can be a later auxiliary technique, but it does not solve spatial conditioning. Do not reactivate it unchanged. | [`pushforward_smoke_20260726.md`](pushforward_smoke_20260726.md), [`long_rollout_architecture_review_20260727.md`](long_rollout_architecture_review_20260727.md) |
| Unconstrained stock residuals plus post-hoc projection | Neural rollout produced negative `carbon_32l`, `DOC`, and `deepC_peat`, causing NaNs and extreme soil-carbon artifacts. Projection removed the immediate numerical failure but did not establish correct transfer ownership. | Inventories must be updated from bounded, source-backed inputs, outputs, and transfers so nonnegativity and conservation hold by construction. Clipping is a guard, not the final architecture. | [`physical_domain_projection_20260726.md`](physical_domain_projection_20260726.md), [`carbon_budget_ownership_audit_20260731.md`](carbon_budget_ownership_audit_20260731.md) |
| Structured process FiLM conditioning | Small spatial gains came with larger seen-point regressions. Parameter/static permutation tests showed weak or wrongly learned conditional use. | Adding parameter channels or FiLM layers does not prove that the network learned the physical parameter response. Conditioning needs explicit intervention tests and later Gate D2 gradient validation. | [`structured_architecture_ab_result_20260726.md`](structured_architecture_ab_result_20260726.md) |
| Mixed-horizon stability objective | Thirty-day aggregate state RMSE improved, but signed daily tendency bias and key carbon variables regressed; 114 of 223 relative gates failed. | Longer-horizon loss is not sufficient. Preserve one-day process supervision, unbiased daily tendencies, and named scientific-field gates. | [`rollout_stability_screening_20260731.md`](rollout_stability_screening_20260731.md) |
| Causal-carbon residual adapter | Numerical, restart, and hard-state gates passed, but direct stock-interface errors for `carbon_32l` and `deepC_peat` worsened by several-fold. Removing those residuals repaired some splits but left severe spatial error. | Direct neural ownership of these stocks remains rejected. Predict source-backed fluxes or transfer fractions and derive stocks through the conservative updater. | [`causal_carbon_adapter_screening_20260731.md`](causal_carbon_adapter_screening_20260731.md), [`causal_carbon_adapter_stock_ablation_result_20260731.md`](causal_carbon_adapter_stock_ablation_result_20260731.md) |
| Neural drivers feeding the exact 48-step `OK_LEAK` scan | The path is executable and its persisted replay is a strong Oracle, but inference still performs 48 recurrent half-hour transitions. | Keep this path for Teacher attribution and Oracle comparisons. It does not satisfy the true-daily product requirement and must not become the Gate E implementation. | [`carbon_budget_ownership_audit_20260731.md`](carbon_budget_ownership_audit_20260731.md), [`conservative_daily_process_operator_v1.md`](conservative_daily_process_operator_v1.md) |

## Reusable Assets

The rejected model candidates do not invalidate the surrounding
infrastructure. The following assets remain reusable when their current
contract and provenance checks pass:

- v5 dataset readers, frozen spatial/temporal splits, streaming statistics,
  batching, checkpoints, and experiment provenance;
- native six-hour forcing records, timestamps, durations, and masks;
- the frozen PFT-axis parameter/trait/mask interface;
- Gate C daily flux-label inventory, supplemental capture, conservative
  updater, and true-label replay cases;
- exact retained-tail code and exact 48-step paths as Teacher/Oracle assets;
- fieldwise diagnostics, free-rollout evaluation, restart-split checks, hard
  state-validity gates, and sealed-test enforcement;
- source-backed parameter registry and Gate D1 Teacher-gradient evidence.

Historical neural model classes and launchers remain reproducibility assets,
not inheritance bases for Gate E1. A passing unit test for one of those files
means the old experiment is reproducible; it does not make the architecture
accepted.

## Reopening Rule

A rejected idea may be reopened only when all of the following are written
before training:

1. A materially new mechanism explains why the prior failure should change.
2. A matched control and bounded A/B gate identify the expected improvement.
3. Named scientific fields, tendency bias, masks, conservation, restart, and
   held-out spatial/temporal slices remain explicit acceptance criteria.
4. The sealed test split is not used for architecture selection.
5. Failure stops the arm; promising training loss alone cannot authorize more
   epochs, wider models, weight tuning, or a full production screen.

Gate E1 is exempt from neural accuracy claims because it is a local executable
skeleton. It must nevertheless obey these negative design constraints before
Gate E2 is allowed to train candidate networks.
