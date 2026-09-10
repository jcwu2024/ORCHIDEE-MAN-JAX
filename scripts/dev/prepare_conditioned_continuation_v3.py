"""Materialize the next forty blocks of the existing seeded training schedule."""
import json
from pathlib import Path

from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_persistent_scale_training import (
    build_persistent_scale_training_schedule,
)


def main():
    root = Path(__file__).resolve().parents[2]
    source = root / 'manifests/coarse_graining/conditioned_snow_free_continuation_v2.json'
    plan = json.loads(source.read_text())
    schedule = build_persistent_scale_training_schedule(seed=20260910, maximum_updates=60, macro_batch_size=40)
    plan.update(schema_version='conditioned_snow_free_continuation_v3', starting_logical_update=34,
                logical_updates=list(range(35, 75)), milestones=[39, 44, 54, 74], save_validation_trajectories=True,
                parent_recovery='/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/training/conditioned-snow-free-continuation-v2-14984032/recovery.pkl',
                parent_report='/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/training/conditioned-snow-free-continuation-v2-14984032/report.json')
    plan['train_panels'] = {str(u): [list(block[i:i + 8]) for i in range(0, 40, 8)]
                            for u, block in zip(range(35, 75), schedule.groups[20:], strict=True)}
    plan['training_schedule']['preceding_blocks'] = 20
    plan['execution'].update(wall_hours=12, worst_case_cny=33.12)
    plan['independent_validation']['interpretation'] = 'reused development validation; excluded from gradients and scale selection; not a fresh or sealed test'
    target = source.with_name('conditioned_snow_free_continuation_v3.json')
    target.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    used = [v for groups in plan['train_panels'].values() for g in groups for v in g]
    print(f'updates=40 artifact_exposures={len(used)} distinct_artifacts={len(set(used))}')


if __name__ == '__main__':
    main()
