# Documentation Status

Last updated: 2026-09-01.

Use this page to decide what must be read and what is historical evidence.

## Authority Order

1. `START_HERE.md`: entry point and reading order.
2. `NEXT_STEPS.md`: current work order and acceptance gates.
3. `CODE_MAP.md`: code ownership and active/historical classification.
4. `current-status.md`: detailed accepted facts and experiment chronology.
5. Named contracts and architecture decisions: durable interface rules.
6. Dated reports: immutable evidence under their original commit and data.

When a dated report conflicts with `NEXT_STEPS.md`, the dated report describes
an earlier experiment and does not control current work.

## Active Documents

| Document | Status | Read when |
| --- | --- | --- |
| `NEXT_STEPS.md` | Current operation authority | Always |
| `CODE_MAP.md` | Current repository map | Always |
| `current-status.md` | Detailed status authority | Looking up accepted evidence |
| `branch-alignment-20260802.md` | Completed Gate A audit | Checking branch provenance |
| `teacher-branch-parity-acceptance.md` | Accepted Gate A decision | Checking canonical Teacher identity |
| `porting/pft14_extensible_design.md` | Accepted Gate B structural design; new PFT science remains unsupported | Extending Teacher PFT support |
| `research/daily_coarse_graining/conservative_daily_process_operator_v1.md` | Active architecture; non-neural boundary accepted | Designing the daily neural model |
| `research/daily_coarse_graining/daily_pft_interface_v1.md` | Frozen Gate B2 contract | Assembling PFT-axis inputs or adding parameter/trait channels |
| `research/daily_coarse_graining/daily_flux_label_inventory_v1.md` | Frozen Gate C1 contract | Reading daily label ownership |
| `research/daily_coarse_graining/gate_c2_constrained_replay_v1.md` | Accepted Gate C2 evidence | Implementing the conservative daily operator |
| `research/daily_coarse_graining/gate_d1_physical_parameter_gradients_v1.md` | Accepted canonical-Teacher gradient evidence | Adding physical parameters or planning inversion |
| `research/daily_coarse_graining/physical_parameter_candidate_registry_v1.md` | Accepted source-driven candidate inventory | Selecting sensitivity or inversion parameters |
| `research/daily_coarse_graining/gate_e1_daily_operator_skeleton.md` | Accepted Gate E1 implementation evidence | Using the true-daily operator boundary |
| `research/daily_coarse_graining/gate_e1_architecture_data_readiness_review.md` | Accepted Gate E1 architecture/data decision | Starting Gate E2 data preparation or candidate admission |
| `research/daily_coarse_graining/coherent_teacher_dataset_generation_v1.md` | Accepted Gate E2 v6 production contract | Reproducing or auditing formal Teacher training data |
| `research/daily_coarse_graining/coherent_teacher_pilot_20260806.md` | Accepted bounded production evidence | Sizing or reviewing coherent Teacher production |
| `research/daily_coarse_graining/teacher_v6_full_production_acceptance_20260807.md` | Accepted full v6 data evidence | Starting or auditing neural training |
| `research/daily_coarse_graining/storage_cleanup_20260807.md` | Current external-asset archive map | Locating or restoring historical v5 workers |
| `research/daily_coarse_graining/failed_architecture_lessons.md` | Active negative-design authority | Selecting or reviewing a neural architecture or objective |
| `research/daily_coarse_graining/first_neural_process_head_candidate_v1.md` | First matched seed completed; comparison closed without promotion | Reviewing the first formal neural process-head design and outcome |
| `research/daily_coarse_graining/first_neural_process_head_smoke_acceptance_20260808.md` | Accepted local and real-v6 pre-training smoke | Reviewing candidate admission or planning the next bounded gate |
| `research/daily_coarse_graining/first_neural_process_training_preflight_20260808.md` | Accepted historical preflight; first pair later completed and the remaining array was stopped | Reviewing timing, cost, or preflight provenance |
| `research/daily_coarse_graining/first_neural_process_head_first_pair_decision_20260808.md` | Controlling stop decision after one matched pair | Reviewing why neither attention nor factorized arm is being continued |
| `research/daily_coarse_graining/long_horizon_biomass_productivity_objective_v1.md` | Active source ledger and metric foundation; its equal-field anchor policy is superseded | Tracing GPP/respiration/biomass source ownership or trajectory extraction |
| `research/daily_coarse_graining/target_oriented_daily_operator_contract_v1.md` | Current frozen loss, output hierarchy, carbon identities, and absolute 7/30/365-day acceptance authority | Discussing or implementing the next successor architecture and training objective |
| `research/daily_coarse_graining/target_oriented_attention_v1.md` | Current selected architecture; local and zero-update real-v6 GPU admission accepted | Reviewing the 83.8M candidate mechanism, output capacity, or bounded-training entry |
| `research/daily_coarse_graining/target_oriented_attention_gpu_admission_20260821.md` | Accepted zero-update real-v6 GPU admission; bounded training design is next | Reviewing gradients, budgets, timing, device memory, or training admission |
| `research/daily_coarse_graining/target_oriented_bounded_training_v1.md` | Current frozen diverse-batch training design; update-250 stage completed and admitted continuation review | Reviewing sampling, optimizer, clipping calibration, milestones, or formal training lifecycle |
| `research/daily_coarse_graining/target_oriented_state_anchor_amendment_20260821.md` | Current frozen loss-conditioning amendment; its representative probe removed DOC domination but exposed a separate reverse-mode defect | Reviewing state-scale fallback, residual `other`, masked gradients, or declared-dynamic `rveget` status |
| `research/daily_coarse_graining/target_oriented_representative_gradient_probe_20260823.md` | Accepted repaired representative gradient gate, including prior negative probes, zero-stock diagnosis, job 14651381 timeout, and passing job 14652590 | Reviewing the NaN repair, source-gradient conditioning, or entry to the eight-batch preflight |
| `research/daily_coarse_graining/target_oriented_full_step_preflight_acceptance_20260824.md` | Accepted eight-batch gradient calibration, resource measurement, and physical preflight used by the completed update-250 stage | Reviewing job 14659152, the frozen p90 clip threshold, hot-step cost, or training entry |
| `research/daily_coarse_graining/target_oriented_formal_training_admission_20260825.md` | Accepted runner/admission and job 14664525 submission provenance; the job is complete | Reviewing prepared assets, recovery admission, or paid resource identity |
| `research/daily_coarse_graining/target_oriented_formal_training_250_result_20260825.md` | Accepted parent-stage result: 28.48% pacing improvement and update-251-to-1000 continuation admitted; superseded as current result by update 1000 | Reviewing the parent checkpoint or why continuation was admitted |
| `research/daily_coarse_graining/target_oriented_update_1000_continuation_v1.md` | Completed continuation, checkpoint-handoff, free-admission, preparation, and update-1000 gate contract | Reviewing the executed updates 251-1000 protocol |
| `research/daily_coarse_graining/target_oriented_formal_training_1000_result_20260826.md` | Current formal training result: slow-carbon breakthrough, 50.56% pacing improvement, absolute flux gate failed | Reviewing update-1000 science or why unchanged continuation is closed |
| `research/daily_coarse_graining/target_oriented_flux_gradient_variance_audit_v1.md` | Completed frozen zero-update diagnostic contract | Reviewing the preregistered batch-size and flux-geometry method |
| `research/daily_coarse_graining/target_oriented_flux_gradient_variance_audit_result_20260826.md` | Accepted diagnostic result: mixed batch and objective geometry; batch-only continuation rejected | Reviewing the evidence against batch-only continuation |
| `research/daily_coarse_graining/target_oriented_component_resolved_flux_basis_v1.md` | Completed frozen amendment and zero-update geometry contract | Reviewing the component basis or audit method |
| `research/daily_coarse_graining/target_oriented_component_resolved_geometry_result_20260826.md` | Accepted geometry evidence; raw mean rejected and the now-completed protected virtual-step gate opened | Reviewing why raw averaging was closed |
| `research/daily_coarse_graining/target_oriented_flux_protected_virtual_step_v1.md` | Completed frozen zero-update direct-step contract | Reviewing the protected-step method or gates |
| `research/daily_coarse_graining/target_oriented_flux_protected_virtual_step_result_20260826.md` | Accepted result; raw amended-loss realization closed and the completed residual-tail audit opened | Reviewing why direct realization failed |
| `research/daily_coarse_graining/target_oriented_flux_residual_tail_audit_v1.md` | Completed frozen zero-training window-loss/gradient-tail contract | Reviewing the preregistered tail and influence method |
| `research/daily_coarse_graining/target_oriented_flux_residual_tail_result_20260826.md` | Accepted result; Jacobian amplification identified and robust-loss-only continuation not admitted | Reviewing why gradient-path attribution was required |
| `research/daily_coarse_graining/target_oriented_flux_gradient_path_attribution_v1.md` | Completed frozen zero-training free-rollout versus Teacher-state gradient-path contract | Reviewing the preregistered attribution method |
| `research/daily_coarse_graining/target_oriented_flux_gradient_path_result_20260826.md` | Current accepted result; scenario-dependent snowheat recurrence identified as the extreme-gradient path | Designing or reviewing the active snowheat recurrence intervention |
| `research/daily_coarse_graining/target_oriented_snowheat_recurrence_intervention_v1.md` | Completed frozen zero-training snowheat-only cross-day causal intervention | Reviewing the preregistered intervention method |
| `research/daily_coarse_graining/target_oriented_snowheat_recurrence_result_20260827.md` | Current accepted result; 7-day snowheat causality confirmed and 30-day residual transferred to snowtemp | Designing or reviewing the active coupled snow-thermal intervention |
| `research/daily_coarse_graining/target_oriented_snow_thermal_recurrence_intervention_v1.md` | Completed frozen two-window snowheat-plus-snowtemp causal intervention | Reviewing the preregistered coupled method |
| `research/daily_coarse_graining/target_oriented_snow_thermal_recurrence_result_20260827.md` | Current accepted coupled result; snowtemp path confirmed but full reduction mixed | Reviewing the coupled result or residual snowheat head |
| `research/daily_coarse_graining/target_oriented_snow_thermal_alias_recurrence_intervention_v1.md` | Completed frozen mirror-complete two-window intervention | Reviewing the preregistered alias recurrence method |
| `research/daily_coarse_graining/target_oriented_snow_thermal_alias_recurrence_result_20260827.md` | Accepted mixed result; mirror snowheat path confirmed and residual moved to direct maintenance | Reviewing the completed alias result or maintenance source audit |
| `research/daily_coarse_graining/target_oriented_maintenance_state_recurrence_intervention_v1.md` | Completed frozen persisted-resp_maint two-window intervention | Reviewing the preregistered maintenance-state method |
| `research/daily_coarse_graining/target_oriented_maintenance_state_recurrence_result_20260827.md` | Current accepted negative result; persisted maintenance recurrence is not explanatory | Reviewing why state-by-state Teacher carry is closed or designing the maintenance-head conditioning audit |
| `research/daily_coarse_graining/target_oriented_maintenance_head_conditioning_audit_v1.md` | Completed frozen zero-training 769-parameter final-head conditioning audit | Reviewing the preregistered method and exact protection gate |
| `research/daily_coarse_graining/target_oriented_maintenance_head_conditioning_result_20260827.md` | Current accepted result; common head direction exists but no finite step is fully protected | Reviewing structured maintenance output parameterization before architecture discussion |
| `research/daily_coarse_graining/target_oriented_source_shaped_maintenance_v1.md` | Frozen source mechanism; branch-only training completed and scientific protection failed | Reviewing the source-shaped head or its coupling failure |
| `research/daily_coarse_graining/target_oriented_source_shaped_maintenance_gpu_admission_20260827.md` | Accepted zero-update real-v6 GPU admission | Reviewing checkpoint transformation, gradients, budgets, memory, or screen entry |
| `research/daily_coarse_graining/target_oriented_source_shaped_train_train_screen_v1.md` | Completed frozen train/train screen contract; job 14701638 executed it unchanged | Reviewing data separation, branch-only optimizer, or the bounded science gate |
| `research/daily_coarse_graining/target_oriented_source_shaped_zero_panel_result_20260827.md` | Accepted zero-training result: large maintenance gain and failed short-horizon protection | Reviewing why bounded branch learning is supported but the zero correction is not accepted |
| `research/daily_coarse_graining/target_oriented_source_shaped_one_update_admission_20260827.md` | Accepted optimizer admission; direct branch-autodiff prerequisite subsequently passed | Reviewing clipping, frozen-parent proof, or one-step safety |
| `research/daily_coarse_graining/target_oriented_source_shaped_branch_autodiff_admission_20260827.md` | Current accepted engineering result; equivalent direct branch gradient with lower hot time and memory | Reviewing the paid 100-update prerequisite or resource envelope |
| `research/daily_coarse_graining/target_oriented_source_shaped_branch_training_admission_20260827.md` | Completed paid-stage admission; job 14701638 executed the accepted envelope | Reviewing implementation, resources, or submission provenance |
| `research/daily_coarse_graining/target_oriented_source_shaped_branch_training_result_20260827.md` | Current completed result: maintenance breakthrough retained, coupled protected science failed | Reviewing why branch-only continuation is closed or preparing source-coupling attribution |
| `research/daily_coarse_graining/target_oriented_source_shaped_coupling_attribution_v1.md` | Completed zero-training causal attribution contract | Reviewing the matched intervention method |
| `research/daily_coarse_graining/target_oriented_source_shaped_coupling_attribution_result_20260828.md` | Current accepted result: protected regressions require recurrence; direct maintenance reverses on Teacher states | Reviewing why joint flux expansion is closed or preparing the source-owner state bridge |
| `research/daily_coarse_graining/target_oriented_source_owner_state_bridge_v1.md` | Completed frozen zero-training four-owner bridge contract | Reviewing the matched three-mode method |
| `research/daily_coarse_graining/target_oriented_source_owner_state_bridge_result_20260828.md` | Accepted result: four direct owners repair part of biomass recurrence but are insufficient for growth/NPP | Reviewing the evidence that motivated the active distribution-aware adapter |
| `research/daily_coarse_graining/target_oriented_distribution_aware_maintenance_v1.md` | Frozen mechanism with accepted local/real-v6 admission and completed bounded training | Reviewing the 24-coordinate Attention adapter or its completed mechanism evidence |
| `research/daily_coarse_graining/target_oriented_distribution_aware_maintenance_gpu_admission_20260828.md` | Accepted zero-update real-v6 GPU admission | Reviewing gradients, parent equality, budgets, timing, or memory evidence |
| `research/daily_coarse_graining/target_oriented_distribution_aware_guard_calibration_v1.md` | Completed frozen zero-update Teacher-guard coefficient protocol | Reviewing the two calibrated objectives |
| `research/daily_coarse_graining/target_oriented_distribution_aware_guard_calibration_result_20260828.md` | Accepted free-GPU result; Teacher-guard coefficient frozen at 0.1 | Reviewing calibration norms, timing, memory, or report identity |
| `research/daily_coarse_graining/target_oriented_distribution_aware_bounded_training_v1.md` | Completed frozen 250-update adapter-only contract; job 14856147 executed it unchanged | Reviewing data separation, optimizer, or the completed science gate |
| `research/daily_coarse_graining/target_oriented_distribution_aware_one_update_admission_20260828.md` | Accepted free-GPU optimizer admission; no update retained | Reviewing combined gradients, clipping, parent preservation, timing, or memory |
| `research/daily_coarse_graining/target_oriented_distribution_aware_training_admission_20260828.md` | Completed paid-stage admission; job 14856147 executed the accepted envelope | Reviewing the exact runner, persistence, resources, timing, or cost |
| `research/daily_coarse_graining/target_oriented_distribution_aware_training_result_20260829.md` | Current completed result: maintenance mechanism passed, full science promotion failed | Reviewing why unchanged adapter continuation is closed or preparing the Teacher-GPP bridge |
| `research/daily_coarse_graining/target_oriented_distribution_aware_gpp_bridge_v1.md` | Completed frozen zero-training Teacher-GPP causal contract | Reviewing the matched modes and preregistered classifier |
| `research/daily_coarse_graining/target_oriented_distribution_aware_gpp_bridge_result_20260829.md` | Accepted causal result: exact GPP repairs all required 7/30-day carbon keys | Reviewing why the active successor changes GPP only |
| `research/daily_coarse_graining/target_oriented_structured_gpp_v1.md` | Current frozen structured-GPP architecture; local, real-v6, direct warm-start, and joint-gradient calibration accepted; retained joint updates closed | Reviewing the 45-token Transformer mechanism, parameter ownership, or bounded gates |
| `research/daily_coarse_graining/target_oriented_structured_gpp_local_acceptance_20260829.md` | Accepted local implementation evidence | Reviewing exact counts, local gradients, PFT behavior, or canonical local budget evidence |
| `research/daily_coarse_graining/target_oriented_structured_gpp_gpu_admission_20260829.md` | Accepted zero-update real-v6 GPU evidence | Reviewing real-v6 forward, gradients, ownership, or daily budget evidence |
| `research/daily_coarse_graining/target_oriented_structured_gpp_training_preflight_v1.md` | Completed frozen specialist-only gradient calibration and discarded-update contract | Reviewing the admitted direct objective, horizon-forward boundary, or optimizer probe |
| `research/daily_coarse_graining/target_oriented_structured_gpp_training_preflight_result_20260829.md` | Accepted free-GPU preflight result; p90 direct-gradient clip frozen | Reviewing gradient coverage, one-update behavior, timing, memory, or report identity |
| `research/daily_coarse_graining/target_oriented_structured_gpp_warm_start_v1.md` | Completed frozen 250-update direct-GPP warm-start contract; job 14873550 executed it unchanged | Reviewing the direct objective, artifact separation, optimizer, endpoint gate, or paid envelope |
| `research/daily_coarse_graining/target_oriented_structured_gpp_warm_start_result_20260829.md` | Accepted capability result: aggregate direct-GPP loss reduced 32.7555% with 8/8 artifact improvements | Reviewing training dynamics, endpoint evidence, checkpoint identity, or why 7/30 calibration is next |
| `research/daily_coarse_graining/target_oriented_structured_gpp_joint_gradient_calibration_v1.md` | Completed frozen zero-update train-only 7/30-day gradient calibration | Reviewing gradient-ratio targets, coefficient aggregation, or preregistered gates |
| `research/daily_coarse_graining/target_oriented_structured_gpp_joint_gradient_calibration_result_20260829.md` | Accepted free-GPU result; 7/30 coefficients and combined p90 clip frozen | Reviewing gradient coverage, positive source geometry, timing, memory, or report identity |
| `research/daily_coarse_graining/target_oriented_structured_gpp_joint_update_preflight_v1.md` | Completed frozen discarded inherited-AdamW proposal | Reviewing the preregistered realized-step gate |
| `research/daily_coarse_graining/target_oriented_structured_gpp_joint_update_preflight_result_20260829.md` | Current rejection evidence; single-artifact paid joint training closed | Reviewing the 30-day conflict, realized step, or why effective batch size is next |
| `research/daily_coarse_graining/target_oriented_structured_gpp_joint_gradient_heterogeneity_audit_v1.md` | Completed frozen zero-update effective-batch protocol | Reviewing artifacts, grouping, or preregistered classification |
| `research/daily_coarse_graining/target_oriented_structured_gpp_joint_gradient_heterogeneity_result_20260829.md` | Accepted result; batch size 4 is the minimum consistent effective batch | Reviewing grouped geometry or why accumulated AdamW is next |
| `research/daily_coarse_graining/target_oriented_structured_gpp_joint_batch4_preflight_v1.md` | Completed frozen discarded batch-four AdamW protocol | Reviewing the two independent proposals or aggregate gates |
| `research/daily_coarse_graining/target_oriented_structured_gpp_joint_batch4_preflight_result_20260830.md` | Accepted free-GPU result; bounded batch-four training design is next | Reviewing realized-step evidence, report-only individual changes, or preparing the paid contract |
| `research/daily_coarse_graining/target_oriented_structured_gpp_joint_training_v1.md` | Completed frozen batch-four contract; paid job 14878155 executed it unchanged | Reviewing schedule, accumulation, recovery, or train-only epoch selection |
| `research/daily_coarse_graining/target_oriented_structured_gpp_joint_training_result_20260830.md` | Accepted training-stage capability; independent scientific screen subsequently failed | Reviewing execution, train-only improvements, checkpoint identity, timing, or cost |
| `research/daily_coarse_graining/target_oriented_structured_gpp_short_horizon_screen_result_20260830.md` | Current failed-promotion evidence and broad-data audit authority | Reviewing the 96-window result, GPP generalization failure, data exposure, or next gate |
| `research/daily_coarse_graining/target_oriented_structured_gpp_data_coverage_result_20260830.md` | Accepted target-blind exact-coverage and monitor authority inherited by the active broad preflight | Reviewing exact exposure, the proposed monitor, or broad-training scale |
| `research/daily_coarse_graining/target_oriented_structured_gpp_streaming_throughput_result_20260830.md` | Accepted direct-only supply and conservative throughput evidence inherited by the active broad preflight | Reviewing the GPP-only data path, throughput, memory, or resource bound |
| `research/daily_coarse_graining/target_oriented_structured_gpp_broad_preflight_v1.md` | Completed frozen preflight contract; schedule and free-GPU result accepted | Reviewing the million-day design or preflight acceptance |
| `research/daily_coarse_graining/target_oriented_structured_gpp_broad_preflight_result_20260830.md` | Accepted target-blind schedule and free-GPU preflight evidence | Reviewing schedule balance, gradients, finite steps, throughput, or clip calibration |
| `research/daily_coarse_graining/target_oriented_structured_gpp_broad_training_v1.md` | Completed frozen 1,000-update stage-1 contract; job 14883081 executed it unchanged | Reviewing the broad-data optimizer, monitor, stopping, recovery, or cost boundary |
| `research/daily_coarse_graining/target_oriented_structured_gpp_broad_training_admission_result_20260830.md` | Accepted free-GPU runner and recovery qualification evidence | Reviewing the two chained updates, physical gate, retained parent, memory, or next paid gate |
| `research/daily_coarse_graining/target_oriented_structured_gpp_broad_training_launch_failure_20260831.md` | Completed launcher-failure and correction authority | Reviewing job 14881105, the empty-array failure, or commit c24f21b |
| `research/daily_coarse_graining/target_oriented_structured_gpp_broad_training_result_20260831.md` | Current accepted broad-stage evidence and next-screen authority | Reviewing job 14883081, monitor improvement, selected checkpoint, bias, physical gates, or the fresh 7/30-day next gate |
| `research/daily_coarse_graining/target_oriented_structured_gpp_fresh_screen_v1.md` | Completed frozen independent 7/30-day screen contract | Reviewing fresh selection, consumed-panel exclusion, direct-GPP diagnostic, or absolute science gates |
| `research/daily_coarse_graining/target_oriented_structured_gpp_fresh_screen_result_20260831.md` | Completed failed-promotion evidence inherited by the attribution result | Reviewing the fresh 96-window result, GPP generalization failure, or downstream flux failures |
| `research/daily_coarse_graining/target_oriented_structured_gpp_fresh_attribution_v1.md` | Completed frozen three-mode zero-training attribution contract | Reviewing local-versus-feedback GPP decomposition, Teacher-GPP bridge causality, integrity gates, or execution boundary |
| `research/daily_coarse_graining/target_oriented_structured_gpp_fresh_attribution_result_20260831.md` | Accepted predecessor attribution; current action is superseded by the source-component scope result | Reviewing the `5/6` direct failures, `6/6` material feedback cells, or Teacher-GPP downstream sufficiency |
| `research/daily_coarse_graining/target_oriented_structured_gpp_state_feedback_attribution_v1.md` | Completed frozen zero-training train-only source-component scope contract | Reviewing the six component masks, updates 301/601/901, GPP capture fraction, materiality threshold, or execution boundary |
| `research/daily_coarse_graining/target_oriented_structured_gpp_state_feedback_attribution_result_20260831.md` | Accepted scope result; carbon selected, water monitored, coupled implementation accepted locally | Reviewing the `96.06%/81.95%` carbon capture, component exclusions, integrity, or coupled-design evidence |
| `research/daily_coarse_graining/target_oriented_coupled_gpp_carbon_feedback_architecture_review_v1.md` | User-approved architecture; local implementation accepted, training unauthorized | Reviewing carbon-coordinate Attention, GPP-only output ownership, train-only objective, matched diagnostic control, or bounded stop rule |
| `research/daily_coarse_graining/target_oriented_coupled_gpp_carbon_feedback_local_acceptance_20260831.md` | Accepted local zero-training implementation; real-v6 admission completed separately | Reviewing coordinate layout, exact parameter ownership, gradients, PFT behavior, or local daily-budget evidence |
| `research/daily_coarse_graining/target_oriented_coupled_gpp_carbon_feedback_gpu_admission_20260901.md` | Accepted real-v6 zero-update admission; reachability subsequently accepted | Reviewing job 14897461, real-v6 gradients, parameter routes, complete transition, budgets, timing, memory, or predecessor evidence |
| `research/daily_coarse_graining/target_oriented_gpp_state_reachability_v1.md` | Completed frozen zero-training dependency-audit contract | Reviewing the 13-day panel, exact-nonzero reachability rule, field grouping, Jacobian gates, or execution boundary |
| `research/daily_coarse_graining/target_oriented_gpp_state_reachability_result_20260901.md` | Accepted reachability evidence; its exact 41-coordinate/16-field inventory is frozen into the current calibration | Reviewing job 14899147, the reachable inventory, sensitivity scales, or physical gates |
| `research/daily_coarse_graining/target_oriented_coupled_gpp_objective_calibration_v1.md` | Completed frozen four-source zero-update calibration contract | Reviewing the reachable-state anchor, gradient sources, trainable boundary, or coefficient rule |
| `research/daily_coarse_graining/target_oriented_coupled_gpp_objective_calibration_result_20260901.md` | Accepted calibration evidence; optimizer realization remains unproven | Reviewing job 14899511, resolved coefficients, clip threshold, artifact-level conflicts, or the reason training remains closed |
| `research/daily_coarse_graining/target_oriented_coupled_gpp_optimizer_preflight_v1.md` | Completed frozen `4e-6` optimizer-proposal contract; finite step rejected | Reviewing fresh-AdamW ownership, eight-artifact aggregation, or predecessor gates |
| `research/daily_coarse_graining/target_oriented_coupled_gpp_optimizer_preflight_result_20260901.md` | Accepted negative finite-step evidence; common first-order direction retained | Reviewing job 14902356, source regressions, direction cosines, physical integrity, or why clipping is not causal |
| `research/daily_coarse_graining/target_oriented_coupled_gpp_optimizer_backoff_v1.md` | Completed frozen exact-half-rate discarded proposal contract | Reviewing the `2e-6` choice, unchanged objectives/data/gates, or execution boundary |
| `research/daily_coarse_graining/target_oriented_coupled_gpp_optimizer_backoff_result_20260901.md` | Accepted optimizer-realization evidence; bounded training contract is now frozen | Reviewing job 14902878, four-source improvements, or the selected `2e-6` step |
| `research/daily_coarse_graining/target_oriented_coupled_gpp_bounded_training_v1.md` | Completed frozen schedule and stop-rule contract; paid stage stopped at update 2 | Reviewing data separation, recovery ownership, or the executed finite-step gate |
| `research/daily_coarse_graining/target_oriented_coupled_gpp_bounded_training_admission_20260902.md` | Accepted free-GPU runner and recovery admission; paid stage subsequently completed | Reviewing the first scheduled macro proposal, runtime, memory, or paid envelope |
| `research/daily_coarse_graining/target_oriented_coupled_gpp_bounded_training_result_20260902.md` | Current completed scientific-stop result; update 1 retained, update 2 rejected, no checkpoint | Reviewing the GPP finite-step regression or designing the active optimizer-geometry audit |
| `research/daily_coarse_graining/target_oriented_coupled_gpp_update2_geometry_audit_v1.md` | Completed corrected zero-training contract; moment rotation diagnosed | Reviewing inherited-moment versus finite-curvature method or the bounded scale grid |
| `research/daily_coarse_graining/target_oriented_coupled_gpp_update2_geometry_result_20260902.md` | Current accepted optimizer diagnosis; reset full step passes, inherited direction fails | Reviewing why fresh moments are the active bounded-training amendment |
| `research/daily_coarse_graining/target_oriented_coupled_gpp_fresh_moment_admission_v1.md` | Completed frozen amendment; free consecutive-update admission executed unchanged | Reviewing fresh-moment ownership, consecutive-update gates, or the no-retention execution boundary |
| `research/daily_coarse_graining/target_oriented_coupled_gpp_fresh_moment_admission_result_20260902.md` | Current scientific rejection; update 2 passes, update 3 exposes raw horizon-direction conflict | Reviewing why fresh moments are insufficient or designing the update-3 direction audit |
| `research/daily_coarse_graining/target_oriented_coupled_gpp_update3_common_direction_audit_v1.md` | Completed frozen zero-retention audit; free execution passed | Reviewing the four-source MGDA direction, matched finite steps, or decision branches |
| `research/daily_coarse_graining/target_oriented_coupled_gpp_update3_common_direction_result_20260902.md` | Current accepted common-direction evidence; all direct scales pass and fresh AdamW is nonmonotonic | Designing the active three-to-five-update direct recalibration smoke |
| `research/daily_coarse_graining/target_oriented_coupled_gpp_direct_recalibration_smoke_v1.md` | Completed frozen three-update smoke; updates 3-4 pass and update 5 rejects the full step | Reviewing repeated recalibration or update-5 finite curvature |
| `research/daily_coarse_graining/target_oriented_coupled_gpp_direct_recalibration_result_20260902.md` | Current two-update transfer evidence and update-5 finite-step rejection | Designing the active bounded update-5 scale audit |
| `research/daily_coarse_graining/target_oriented_coupled_gpp_update5_scale_audit_v1.md` | Current frozen zero-retention update-5 scale audit; local qualification passed | Running or reviewing the bounded finite-scale execution |
| `research/daily_coarse_graining/target_oriented_update_251_admission_20260825.md` | Accepted exact checkpoint/AdamW handoff, update-251, recovery, and matched free-GPU admission | Preparing or admitting updates 251-1000 |
| `research/daily_coarse_graining/daily_operator_parameter_routing_v1.md` | Implemented local parameter-ownership boundary used by the successor | Routing physical parameters or planning Gate D2 |
| `research/daily_coarse_graining/successor_process_head_architecture_options_v1.md` | Historical accepted decision package; its attention/graph pair is not the current successor default | Reviewing the completed early successor comparison |
| `research/daily_coarse_graining/successor_process_head_local_foundation_20260809.md` | Accepted local synthetic foundation; real-v6 lifecycle later exposed an incomplete fast boundary | Reviewing implementation evidence before the boundary diagnosis |
| `research/daily_coarse_graining/successor_fast_boundary_coverage_20260809.md` | Accepted blocking diagnosis; superseded for current action by the owner schema | Tracing why the original 101-output boundary failed |
| `research/daily_coarse_graining/successor_fast_owner_reconstruction_contract_v1.md` | Accepted 160/160 owner/materializer boundary; decoder extension implemented | Auditing the 79-value-head plus mask ownership contract |
| `research/daily_coarse_graining/successor_fast_owner_decoder_local_acceptance_20260809.md` | Accepted local 181-field decoder/statistics foundation | Reviewing decoder semantics or local gates |
| `research/daily_coarse_graining/successor_fast_owner_statistics_acceptance_20260809.md` | Accepted formal v6 79-value statistics artifact | Reviewing statistics identity, counts, or hashes |
| `research/daily_coarse_graining/successor_fast_owner_gpu_probe_acceptance_20260809.md` | Accepted real-v6 GPU execution/gradient gate; bounded-fit preparation is next | Reviewing GPU evidence, threshold policy, or the next gate |
| `research/daily_coarse_graining/successor_small_fit_runner_contract_v1.md` | Active bounded-small-fit contract; free single-update preflight accepted | Changing sampling, objective, optimizer, lifecycle scope, or paid-gate policy |
| `research/daily_coarse_graining/successor_small_fit_gpu_probe_acceptance_20260809.md` | Accepted free GPU replay, coefficient, gradient, and one-update gate | Reviewing bounded-fit evidence or implementing the paid production runner |
| `research/daily_coarse_graining/successor_small_fit_runner_reuse_acceptance_20260810.md` | Historical accepted same-process reuse evidence; its 20-hour paid projection is superseded | Reviewing why repeated host preparation was rejected |
| `research/daily_coarse_graining/successor_small_fit_prepared_batch_acceptance_20260810.md` | Accepted prepared-batch and hot-reuse evidence; cross-process cache and old paid envelope rejected | Reviewing the prepared boundary or rejected cache assumption |
| `research/daily_coarse_graining/successor_small_fit_compile_reduction_acceptance_20260810.md` | Accepted cold/hot timing evidence; avoidable compilation removed | Reviewing the timing basis for the bounded 500-update resource plan |
| `research/daily_coarse_graining/successor_small_fit_production_resource_plan_20260810.md` | Historical accepted staged resource contract; both stages complete | Reviewing resources, costs, or recovery policy |
| `research/daily_coarse_graining/successor_small_fit_prepared_500_acceptance_20260810.md` | Accepted immutable 500-entry CPU input to the completed fit | Reviewing prepared input identity |
| `research/daily_coarse_graining/successor_small_fit_checkpoint_handoff_v1.md` | Active checkpoint amendment; production dual checkpoint accepted | Consuming the checkpoints in rollout |
| `research/daily_coarse_graining/successor_checkpoint_handoff_gpu_probe_acceptance_20260810.md` | Historical accepted free-GPU serialization evidence; no checkpoint was retained by that probe | Reviewing pre-fit persistence evidence |
| `research/daily_coarse_graining/successor_small_fit_paid_admission_failure_20260811.md` | Accepted failed-run diagnosis; schedule-ownership defect closed by the production admission | Reviewing job 14517771 or the reason for the admission repair |
| `research/daily_coarse_graining/successor_production_fit_asset_admission_20260811.md` | Historical accepted exact-production admission used by the completed fit | Reviewing production asset identity |
| `research/daily_coarse_graining/successor_small_fit_500_checkpoint_acceptance_20260811.md` | Active accepted 500-update fit and dual checkpoint product | Preparing or admitting the 7/30-day rollout screen |
| `research/daily_coarse_graining/successor_7_30_rollout_screen_package_20260810.md` | Active frozen post-fit acceptance package; dual checkpoints are now available | Preparing or reviewing the free smoke and formal bounded 7/30-day screen |
| `research/daily_coarse_graining/successor_7_30_free_smoke_acceptance_20260811.md` | Active accepted free-smoke execution evidence; not promotion eligible | Preparing or reviewing the formal 96-window bounded screen |
| `research/daily_coarse_graining/successor_7_30_formal_prepared_acceptance_20260811.md` | Active accepted formal prepared-panel input; paid screen awaits approval | Admitting or reviewing the formal 96-window bounded screen |
| `research/daily_coarse_graining/successor_7_30_formal_cache_timeout_20260811.md` | Accepted infrastructure-failure diagnosis; repair confirmed by the completed free screen | Reviewing why job 14520704 timed out without model evidence |
| `research/daily_coarse_graining/successor_7_30_formal_screen_result_20260812.md` | Accepted formal rejection of the trained attention checkpoint; control not promoted | Reviewing the bounded 7/30-day checkpoint result |
| `research/daily_coarse_graining/successor_training_sufficiency_loss_gradient_audit_20260812.md` | Accepted attribution; its decoder diagnosis remains active evidence and its next-action list is completed | Reviewing data use, loss repair, coefficient instability, or the matched output-head repair |
| `research/daily_coarse_graining/successor_staged_training_assets_acceptance_20260812.md` | Active accepted v3 coefficient and milestone-panel inputs; staged runner and free smoke now pass | Reviewing the expanded matched training inputs |
| `research/daily_coarse_graining/successor_staged_training_gpu_smoke_acceptance_20260812.md` | Active accepted free-GPU staged execution preflight; not promotion evidence | Sizing or reviewing the first 500-update stage |
| `research/daily_coarse_graining/successor_staged_500_milestone_acceptance_20260812.md` | Accepted first staged baseline; superseded for current action by the 1000 milestone | Reviewing update-500 provenance or comparison values |
| `research/daily_coarse_graining/successor_staged_1000_milestone_acceptance_20260813.md` | Active accepted second staged milestone; both arms improve but neither is promoted | Reviewing or proposing the conditional 1001-2000 stage |
| `research/daily_coarse_graining/successor_batch_diversity_probe_20260814.md` | Accepted equal-sample A/B; mixed sampling is not a universal policy | Reviewing sampling diversity and the Attention-specific improvement |
| `research/daily_coarse_graining/successor_loss_conflict_attribution_20260814.md` | Accepted validation attribution; superseded for current action by the gradient audit | Reviewing the maintenance-versus-productivity tradeoff |
| `research/daily_coarse_graining/successor_per_output_gradient_audit_20260814.md` | Active train-only gradient evidence; mixed-conflict mechanism rejected and loss-design review opened | Designing the next matched objective experiment |
| `research/daily_coarse_graining/successor_loss_objective_design_review_v1.md` | Active proposal only; review required before implementation or training | Discussing the named biomass-tendency mechanism and matched gates |
| `research/daily_coarse_graining/successor_gradient_balanced_smoke_20260814.md` | Active rejection evidence; 100-update continuation closed | Reviewing raw-gradient versus realized-AdamW-step failure and the next zero-update audit |
| `research/daily_coarse_graining/successor_realized_step_audit_20260815.md` | Active optimizer-handoff rejection; inherited/reset AdamW closed | Designing the zero-update direct-direction trust-region gate |
| `research/daily_coarse_graining/successor_direct_trust_region_audit_20260815.md` | Active direct-step admission; global norm 0.001 selected | Designing the no-checkpoint 10-update direct-direction smoke |
| `research/daily_coarse_graining/successor_direct_direction_smoke_20260815.md` | Active per-batch direction-transfer rejection; no continuation | Designing the zero-update robust multi-batch direction audit |
| `research/daily_coarse_graining/successor_multibatch_direction_audit_20260815.md` | Active batch-size attribution; ordinary averaging rejected | Designing the group-aware 40-objective feasibility audit |
| `research/daily_coarse_graining/successor_group_aware_direction_audit_20260815.md` | Active positive common-direction evidence; original finite-step grid failed magnitude only | Running or reviewing the bounded zero-update step extension |
| `research/daily_coarse_graining/successor_group_aware_step_extension_20260815.md` | Active finite-step admission; global norm 0.002 selected | Running or reviewing the five-update frozen-scalarization smoke |
| `research/daily_coarse_graining/successor_group_aware_scalarization_smoke_20260815.md` | Active fixed-step drift rejection with positive one-step rollout signal | Running or reviewing the bounded backtracking smoke |
| `research/daily_coarse_graining/successor_group_aware_backtracking_smoke_20260815.md` | Active fixed-weight rejection; smaller steps do not restore common descent | Designing the second-state direction recalibration audit |
| `research/daily_coarse_graining/successor_second_state_recalibration_audit_20260815.md` | Active exact second-state admission; sparse overlapping support accepted | Implementing or reviewing efficient active-set recalibration |
| `research/daily_coarse_graining/successor_active_set_recalibration_audit_20260815.md` | Active efficient recalibration admission; 14-VJP one-round closure accepted | Designing or reviewing the repeated active-set smoke |
| `research/daily_coarse_graining/successor_repeated_active_set_smoke_20260815.md` | Active rejection evidence; one new update passes, then useful one-shot steps close | Designing the cumulative-microstep gate |
| `research/daily_coarse_graining/successor_cumulative_microstep_smoke_20260815.md` | Active rejection evidence; safe microsteps do not accumulate through the finite-step boundary | Designing parameter-geometry attribution |
| `research/daily_coarse_graining/successor_parameter_geometry_attribution_20260815.md` | Active negative attribution evidence; decoder dominates but no top-level exclusion is safe | Designing decoder-leaf attribution |
| `research/daily_coarse_graining/successor_decoder_leaf_attribution_20260816.md` | Active negative decoder attribution; raw Euclidean direct steps closed | Designing output-locality and function-space trust region |
| `research/daily_coarse_graining/successor_output_locality_audit_20260816.md` | Active positive field-local attribution; maintenance output dominates the finite failure | Designing a normalized maintenance-local output trust region |
| `research/daily_coarse_graining/successor_maintenance_output_projection_protocol_20260816.md` | Completed negative evidence; a fixed maintenance gradient cannot jointly restore the caps and all losses | Designing a bounded nonlinear constrained solver in a frozen low-dimensional subspace |
| `research/daily_coarse_graining/successor_low_dimensional_nonlinear_projection_protocol_20260816.md` | Completed negative evidence; the bounded solver found no admissible nonzero step | Reviewing why post-training low-dimensional projection is closed |
| `research/daily_coarse_graining/successor_attention_mixed_annealed_continuation_protocol_20260816.md` | Completed frozen Attention-only training contract; candidate not admitted | Reviewing the preregistered recipe or execution boundary |
| `research/daily_coarse_graining/successor_attention_mixed_annealed_continuation_result_20260816.md` | Active negative training evidence; maintenance improves while every primary 7/30-day cell regresses | Reviewing objective alignment versus backbone/capacity with the user |
| `research/daily_coarse_graining/successor_route_local_decoder_architecture_review_v1.md` | Superseded before implementation; retained unexecuted diagnostic hypothesis | Tracing why decoder isolation was considered |
| `research/daily_coarse_graining/strong_baseline_axis_aligned_process_expert_review_v1.md` | Accepted strong-baseline architecture and staged absolute gates; local implementation complete | Reviewing the approved mechanism or qualification sequence |
| `research/daily_coarse_graining/strong_baseline_local_acceptance_20260816.md` | Accepted local structural evidence; real-v6 admission later passed | Reviewing executable capacity, source/axis wiring, dtype, gradient, or local budget evidence |
| `research/daily_coarse_graining/strong_baseline_one_point_qualification_result_20260816.md` | Active failed qualification and next diagnostic authority | Reviewing one-point fit, expert loss floor, key science metrics, or the blocked 64-point-year gate |
| `research/daily_coarse_graining/strong_baseline_one_point_loss_floor_diagnostic_v1.md` | Completed frozen diagnostic contract; preflight and full execution complete | Reviewing the preregistered method or execution boundary |
| `research/daily_coarse_graining/strong_baseline_one_point_loss_floor_diagnostic_result_20260817.md` | Failed diagnostic and dominant-label attribution evidence | Reviewing why one-point qualification remains closed |
| `research/daily_coarse_graining/dominant_endpoint_source_representation_audit_20260817.md` | Completed corrected endpoint, exact-reconstruction, and DOC-semantics authority | Designing the next primitive-output/exact-reconstruction process head |
| `research/daily_coarse_graining/doc_diffusion_boundary_baseline_result_20260817.md` | Accepted physical DOC boundary baseline plus learned-residual evidence; not an exact-process claim | Implementing or evaluating the revised DOC output after approval |
| `research/daily_coarse_graining/mechanistic_output_architecture_review_v1.md` | Accepted architecture; implementation, real-v6 admission, and one-epoch training smoke complete | Reviewing the active mechanistic-output process head or one-point gate |
| `research/daily_coarse_graining/mechanistic_output_one_point_qualification_result_20260817.md` | Active failed-qualification, positive-mechanism, field-attribution, and training-stability evidence | Designing the next hydrology/canopy representation revision |
| `research/daily_coarse_graining/hydrology_canopy_boundary_audit_result_20260817.md` | Accepted hydrology/canopy source audit and v2 mechanism basis | Reviewing source evidence behind final `nroot` and signed canopy endpoints |
| `research/daily_coarse_graining/hydrology_canopy_output_v2_local_acceptance_20260817.md` | Accepted local v2 evidence; real-v6 admission subsequently passed | Reviewing the v2 implementation and local gate |
| `research/daily_coarse_graining/hydrology_canopy_output_v2_real_v6_admission_20260817.md` | Accepted real-v6 v2 admission; one-epoch smoke subsequently passed | Reviewing the complete forward/loss/full-gradient/state/budget evidence |
| `research/daily_coarse_graining/hydrology_canopy_output_v2_one_epoch_smoke_result_20260817.md` | Completed accepted smoke and historical paid-envelope contract | Reviewing the admission path to the formal run |
| `research/daily_coarse_graining/hydrology_canopy_output_v2_one_point_qualification_result_20260818.md` | Completed scientific rejection and motivation for the accepted attainability audit | Reviewing the v2 training result or why representation attainability was tested |
| `research/daily_coarse_graining/mechanistic_output_v2_attainability_result_20260818.md` | Accepted output-coordinate and thermal-source authority; isolated diagnostics now complete | Reviewing why output-coordinate reachability is not the remaining blocker |
| `research/daily_coarse_graining/mechanistic_output_v2_isolated_expert_memorization_protocol_20260818.md` | Completed frozen bounded training diagnostic | Reviewing the preregistered objective, budget, and gates |
| `research/daily_coarse_graining/mechanistic_output_v2_isolated_expert_memorization_result_20260818.md` | Accepted negative mapping and field-level evidence; revision now implemented locally | Reviewing why the rejected litter mapping was replaced |
| `research/daily_coarse_graining/hierarchical_coordinate_query_attention_v3_design_20260819.md` | Active accepted candidate mechanism and bounded gate contract | Reviewing the litter coordinate-query design or its absolute gates |
| `research/daily_coarse_graining/hierarchical_coordinate_query_attention_v3_local_acceptance_20260819.md` | Accepted local candidate admission; first GPU attempt exposed a runner-memory issue | Reviewing candidate correctness before GPU execution |
| `research/daily_coarse_graining/hierarchical_coordinate_query_attention_v3_gpu_smoke_attempt_20260819.md` | Historical zero-update OOM evidence; the repaired smoke and formal diagnostic later completed | Reviewing the memory repair rationale |
| `research/daily_coarse_graining/hierarchical_coordinate_query_attention_v3_doc_audit_20260820.md` | Accepted failed absolute qualification and DOC coordinate-coupling basis for v4 | Reviewing why v4 was opened |
| `research/daily_coarse_graining/coordinate_coupled_doc_transition_v4_design_20260820.md` | Completed frozen v4 mechanism contract; formal candidate subsequently rejected | Reviewing the coordinate-aligned DOC design |
| `research/daily_coarse_graining/coordinate_coupled_doc_transition_v4_local_acceptance_20260820.md` | Accepted local v4 foundation; free smoke subsequently passed | Reviewing candidate correctness before GPU execution |
| `research/daily_coarse_graining/coordinate_coupled_doc_transition_v4_gpu_smoke_20260820.md` | Accepted free-GPU smoke; formal diagnostic subsequently completed and failed | Reviewing the execution admission path |
| `research/daily_coarse_graining/coordinate_coupled_doc_transition_v4_memorization_result_20260820.md` | Completed failed field-gate evidence; the follow-up geometry audit now controls the next gate | Reviewing why the joint signed v4 head was rejected |
| `research/daily_coarse_graining/doc_endpoint_target_loss_geometry_audit_v1.md` | Completed frozen zero-training protocol | Reviewing the preregistered DOC target/loss diagnostics |
| `research/daily_coarse_graining/doc_endpoint_target_loss_geometry_audit_result_20260820.md` | Accepted joint-head rejection and v5 mechanism basis | Reviewing why status plus branch magnitude replaced the signed scalar |
| `research/daily_coarse_graining/doc_status_branch_magnitude_attention_v5_design_20260820.md` | Completed frozen v5 mechanism contract; early-screen candidate subsequently rejected | Reviewing the tested status/branch-magnitude design |
| `research/daily_coarse_graining/doc_status_branch_magnitude_attention_v5_local_acceptance_20260821.md` | Accepted local foundation; superseded for current action by the completed early screen | Reviewing implementation correctness before GPU execution |
| `research/daily_coarse_graining/doc_status_branch_magnitude_attention_v5_early_screen_result_20260821.md` | Accepted negative evidence; no longer the current objective authority | Reviewing why the v5 physical materialization head is rejected |
| `research/daily_coarse_graining/development_standard.md` | Active policy | Adding a research experiment |
| `deployment-explore1000.md` | Active platform instructions | Running on Explore1000 |
| `installation.md`, `data-layout.md`, `running.md` | Active user docs | Installing or running the Teacher |

## Conditional Contracts

- `research/daily_coarse_graining/daily_markov_contract_v5.md` is active for
  reading existing v5 Teacher shards. It is superseded as the final neural
  boundary by the conservative daily operator decision.
- `research/daily_coarse_graining/teacher_dataset_generation.md` remains the
  production and provenance contract for the historical v4/v5 Teacher data
  product. New formal data use the coherent generation contract above.
- `docs/source_audits/*.yaml`, `oracle_families/`, and
  `production_families/` are machine/evidence contracts used by their audit
  commands. They are not general onboarding material.
- `PROJECT_MANIFEST.md` owns repository and external-asset boundaries.

## Historical Documents

- `research/daily_coarse_graining/HANDOFF.md` is a chronological research log.
  It is preserved for provenance and is no longer the current-operation
  authority.
- Dated `*_202607*.md` and `*_202608*.md` research reports are immutable
  experiment snapshots.
- `research/daily_coarse_graining/gate_e2_typed_sidecar_data_preparation.md`
  records the rejected immutable-parent sidecar design and server attempts.
- `daily_boundary_v0_draft.yaml`, `daily_markov_contract_v3.md`, and
  `daily_markov_contract_v4.md` are superseded contracts retained for
  migration provenance.
- Stage-numbered plans and completion notes under `docs/source_audits/` record
  the PFT14 porting campaign. They do not define the daily-surrogate roadmap.
- Neural launchers and manifests named for rejected `canonical`, `structured`,
  `rollout_stability`, or `causal_carbon` experiments remain reproducibility
  assets, not active defaults.

Do not delete historical material merely to make the tree shorter. Navigate it
through this status index and `CODE_MAP.md`; move or prune only in a dedicated
archive change that preserves links and provenance hashes.
