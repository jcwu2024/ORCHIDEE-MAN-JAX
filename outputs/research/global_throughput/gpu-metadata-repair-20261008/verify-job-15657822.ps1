$ErrorActionPreference = 'Stop'
$Evidence = $PSScriptRoot
$Collected = Get-Content "$Evidence/job-15657822-collected-result.json" -Raw | ConvertFrom-Json
$Report = Get-Content "$Evidence/job-15657822-saved-capacity/report.json" -Raw | ConvertFrom-Json
$Manifest = Get-Content "$Evidence/job-15657822-saved-capacity/stage_manifest.json" -Raw | ConvertFrom-Json
$Stage = $Collected.valid_exploratory_stages[0]
$Record = $Manifest.stages.capacity_B512_fixed512_v1
$Checkpoint = Get-Item "$Evidence/job-15657822-saved-capacity/stage_capacity_B512_fixed512_v1.pkl"
$CheckpointHash = (Get-FileHash $Checkpoint.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
$Checks = [ordered]@{
    collector_exit_zero = ((Get-Content "$Evidence/job-15657822-collection-exit.txt").Trim() -eq '0')
    collector_stderr_empty = ((Get-Item "$Evidence/job-15657822-collection-stderr.txt").Length -eq 0)
    exactly_one_valid_stage_no_withheld = ($Collected.valid_exploratory_stages.Count -eq 1 -and $Collected.withheld_stages.Count -eq 0)
    collector_strict_full_reference_pass = ($Stage.strict_numerical_passed -and $Stage.actual_saved_output_reference_passed)
    all_515_saved_gates_pass = ($Record.gates.Count -eq 515 -and @($Record.gates | Where-Object { -not $_.passed }).Count -eq 0)
    all_512_independent_760_leaf_comparisons_pass = ($Stage.actual_saved_output_reference_comparisons.Count -eq 512 -and @($Stage.actual_saved_output_reference_comparisons | Where-Object { -not $_.passed -or $_.compared_leaves -ne 760 }).Count -eq 0)
    local_checkpoint_hash_matches_manifest_and_collector = ($CheckpointHash -eq $Record.sha256 -and $CheckpointHash -eq $Stage.checkpoint_sha256)
    local_checkpoint_exact_bytes = ($Checkpoint.Length -eq 220430830 -and $Record.checkpoint_io.bytes -eq $Checkpoint.Length)
    report_record_exact_manifest_record = (($Report.stages.capacity_B512_fixed512_v1 | ConvertTo-Json -Depth 100 -Compress) -ceq ($Record | ConvertTo-Json -Depth 100 -Compress))
    completed_single_B512_only = ($Report.status -eq 'completed_exploratory_capacity_screen' -and $Report.completed_shapes.Count -eq 1 -and $Report.completed_shapes[0] -eq 512)
    five_core_and_wrapper_samples = ($Stage.core.samples_seconds.Count -eq 5 -and $Stage.wrapper.samples_seconds.Count -eq 5)
    immutable_code_retained = ((Get-Content "$Evidence/job-15657822-code-after-collection.txt").Trim() -eq 'ad6c0e1faa7e22b590627844e9cb0ad492e9deb3' -and (Get-Item "$Evidence/job-15657822-clean-after-collection.txt").Length -eq 0)
    terminal_parent_completed_1416_seconds = ($Collected.accounting.state -eq 'COMPLETED' -and $Collected.accounting.exit_code -eq '0:0' -and $Collected.accounting.observed_wall_seconds -eq 1416)
    exact_single_raw9_parent = (@(Get-Content "$Evidence/job-15657822-terminal-sacct-raw9.txt" | Where-Object { $_.StartsWith('15657822|') }).Count -eq 1)
    scientific_acceptance_false = (-not $Collected.scientific_acceptance -and -not $Stage.scientific_acceptance)
}
if (@($Checks.Values | Where-Object { -not $_ }).Count) { throw ($Checks | ConvertTo-Json) }
$Summary = [ordered]@{
    status = 'passed'
    job_id = '15657822'
    checks = $Checks
    checkpoint_sha256 = $CheckpointHash
    checkpoint_bytes = $Checkpoint.Length
    core = $Stage.core
    wrapper = $Stage.wrapper
    setup = $Stage.startup
    checkpoint_IO = $Stage.checkpoint_IO
    terminal_accounting = $Collected.accounting
    B256_wrapper_observed_throughput_ratio = 30.468557430 / $Stage.wrapper.median_seconds
    B256_wrapper_warm_cost_reduction_percent = 100 * (1 - $Stage.wrapper.median_seconds / 30.468557430)
    wrapper_CNY_per_million_lane_days_primary = $Stage.wrapper.warm_cost_scenarios.primary.CNY * 1000000 / 7168
    wrapper_CNY_per_million_lane_days_conservative = $Stage.wrapper.warm_cost_scenarios.conservative.CNY * 1000000 / 7168
    max_saved_reference_bound_ratio = ($Stage.actual_saved_output_reference_comparisons | Measure-Object max_bound_ratio -Maximum).Maximum
    collector_CPU_saved_array_wall_seconds = 80.06
    collector_execution_node = 'test02'
    collector_affinity_max = 4
    collector_thread_limits = 'OMP/OPENBLAS/MKL=1'
    collection_budget_scope = 'separate saved-array validation; not added to preparation diagnostics or paid parent allocation wall'
    source_admission = '/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/gpu-sweetspot-preparation-20261006/target-input-admission.json'
    source_admission_use = 'existing independently admitted original FULL64/source identity; strict deployed collector revalidated exact current origins'
    verification_scope = 'local preservation/metadata checks and arithmetic; numerical validation belongs to actual deployed collector on test02'
    limitations = @('exploratory bounded capacity result only', 'historical B128 lazy dispatch differs; separate jobs and repaired cache prevent isolated batch causality', 'no CPU ratio, K amortization, global science, Fortran, annual, 669 tail/IO, 50-year, 10x or gradients acceptance', 'actual compiler memory and selected-card snapshots are not measured VRAM peak', 'cost scenarios are not verified billing')
}
$Summary | ConvertTo-Json -Depth 100 | Set-Content "$Evidence/job-15657822-verification.json" -Encoding utf8
Write-Output "Passed $($Checks.Count) preservation/metadata checks."
