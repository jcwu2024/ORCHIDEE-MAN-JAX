"""Plot actual independent-panel science fits and stock changes in physical units."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NAMES = {
    'gpp': 'GPP 总初级生产力', 'npp': 'NPP 净初级生产力',
    'maintenance_respiration': '维持呼吸', 'growth_respiration': '生长呼吸',
    'autotrophic_respiration': '自养呼吸（维持＋生长）', 'heterotrophic_respiration': '异养呼吸',
    'aboveground_biomass_carbon': 'AGB 地上生物量碳',
    'belowground_biomass_carbon': 'BGB 地下生物量碳',
}
PARENT, CURRENT, TEACHER = '#a2a9b3', '#087f8c', '#222b38'


def values(row, label, category, key):
    return np.asarray(row[label][category][key], dtype=float)


def points(rows, label, category, key, change=False):
    arrays = [values(row, label, category, key) for row in rows]
    if category == 'fluxes':
        return np.asarray([np.mean(a) for a in arrays])
    return np.asarray([a[-1] - a[0] if change else a[-1] for a in arrays])


def metrics(truth, predicted):
    valid = np.isfinite(truth) & np.isfinite(predicted)
    if not valid.all():
        raise ValueError('unexpected undefined value in the plotted science sample')
    error = predicted - truth
    return dict(n=len(truth), teacher_mean=float(np.mean(truth)), predicted_mean=float(np.mean(predicted)),
                rmse=float(np.sqrt(np.mean(error**2))), bias=float(np.mean(error)),
                mae=float(np.mean(np.abs(error))), teacher_min=float(np.min(truth)), teacher_max=float(np.max(truth)))


def scatter(ax, rows, category, key, *, change=False):
    truth = points(rows, 'teacher', category, key, change)
    before = points(rows, 'update14', category, key, change)
    after = points(rows, 'update34', category, key, change)
    lo, hi = min(truth.min(), before.min(), after.min()), max(truth.max(), before.max(), after.max())
    margin = max((hi - lo) * .08, .001)
    ax.plot([lo-margin, hi+margin], [lo-margin, hi+margin], '--', color=TEACHER, lw=1, label='理想：预测＝教师')
    for horizon, marker in ((7, 'o'), (30, '^')):
        index = np.asarray([r['horizon'] == horizon for r in rows])
        ax.scatter(truth[index], before[index], color=PARENT, marker=marker, s=33, alpha=.8)
        ax.scatter(truth[index], after[index], color=CURRENT, marker=marker, s=43, alpha=.85,
                   label=f'当前 {horizon} 日窗口')
        for x, old, new in zip(truth[index], before[index], after[index], strict=True):
            ax.plot([x, x], [old, new], color=CURRENT, lw=.7, alpha=.3)
    ax.set_xlim(lo-margin, hi+margin)
    ax.set_ylim(lo-margin, hi+margin)
    ax.set_aspect('equal', adjustable='box')
    label = NAMES[key] + ('：窗口内变化' if change else '')
    unit = 'gC/m²/天' if category == 'fluxes' else 'gC/m²'
    ax.set(title=label, xlabel=f'教师（{unit}）', ylabel=f'模型（{unit}）')
    m, old = metrics(truth, after), metrics(truth, before)
    ax.text(.035, .965, f'当前 RMSE {m["rmse"]:.3g}\n平均偏差 {m["bias"]:+.3g}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(facecolor='white', edgecolor='none', alpha=.8))
    if np.all(np.abs(truth) < 1e-12) and np.all(np.abs(after) < 1e-12):
        ax.text(.5, .25, '本面板教师与模型均为零\n不能证明非零情景的预测能力',
                ha='center', transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.9))
    ax.grid(alpha=.15)
    return {'current': m, 'parent': old}


def render(source, destination):
    report = json.loads(source.read_text(encoding='utf-8'))
    if report['status'] != 'completed' or not report['report_reproduced']:
        raise ValueError('requires completed export reproducing the saved acceptance report')
    rows = report['windows']
    if len(rows) != 24:
        raise ValueError('plot all 24 independent windows')
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    summary = {'source_job': report['source_job'], 'windows': 24,
               'metric_definition': 'equal-window physical-unit RMSE/bias; flux window means, stocks window endpoints',
               'fluxes': {}, 'stocks': {}, 'stock_changes': {}}
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6), layout='constrained')
    flux_keys = tuple(NAMES)[:6]
    for ax, key in zip(axes.flat, flux_keys, strict=True):
        summary['fluxes'][key] = scatter(ax, rows, 'fluxes', key)
    axes[0, 2].legend(loc='lower right', fontsize=8)
    fig.suptitle('生产力与呼吸：当前预测离教师还有多远？\n每个点是一段独立窗口的日均值；灰色＝续训前，青色＝当前；越靠近虚线越好', fontsize=14)
    fig.savefig(destination / 'flux_fit.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9), layout='constrained')
    for column, key in enumerate(tuple(NAMES)[6:]):
        summary['stocks'][key] = scatter(axes[0, column], rows, 'stocks', key)
        summary['stock_changes'][key] = scatter(axes[1, column], rows, 'stocks', key, change=True)
    fig.suptitle('生物量：总量接近，不代表增长趋势也准确\n上排＝期末碳库；下排＝各自第1日到末日的碳库变化（不是全年增长）', fontsize=14)
    fig.savefig(destination / 'biomass_fit.png', dpi=160)
    plt.close(fig)
    first = rows[0]
    key = 'aboveground_biomass_carbon'
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), layout='constrained')
    top = 0.
    case = {}
    for label, color, name in (('teacher', TEACHER, '教师'), ('update34', CURRENT, '当前模型')):
        a = values(first, label, 'stocks', key)
        top = max(top, float(a.max()))
        case[label] = dict(first_day=float(a[0]), last_day=float(a[-1]), change=float(a[-1]-a[0]))
        axes[0].plot(range(1, len(a)+1), a, color=color, marker='o', label=f'{name}：末日 {a[-1]:.2f}')
        axes[1].plot(range(1, len(a)+1), a-a[0], color=color, marker='o', label=f'{name}：增加 {a[-1]-a[0]:.2f}')
    axes[0].set_ylim(0, top*1.08)
    for ax, title in zip(axes, ('看碳库总量：两条线几乎重合', '看窗口内增加量：差异明显'), strict=True):
        ax.set(title=title, xlabel='窗口内第几天', ylabel='地上生物量碳，gC/m²')
        ax.legend(fontsize=9, loc='upper left' if ax == axes[1] else 'center left')
        ax.grid(alpha=.15)
    fig.suptitle(f'同一个独立7日窗口：{first["entry"]["landpoint_id"]}，{first["entry"]["year"]}\n固定面板首个窗口；增加量从各自第1日算起，不能外推全年', fontsize=13)
    fig.savefig(destination / 'stock_level_vs_growth.png', dpi=170)
    plt.close(fig)
    summary['first_window_stock_example'] = {'entry': first['entry'], **case}
    thirty = [r for r in rows if r['horizon'] == 30]
    ranked = sorted(thirty, key=lambda r: metrics(values(r, 'teacher', 'fluxes', 'gpp'),
                                                values(r, 'update34', 'fluxes', 'gpp'))['rmse'])
    selected = [ranked[0], ranked[len(ranked)//2], ranked[-1]]
    summary['example_selection'] = 'smallest, upper-middle, largest daily GPP RMSE among all twelve 30-day windows'
    summary['example_ordinals'] = [r['ordinal'] for r in selected]
    fig, axes = plt.subplots(3, 4, figsize=(15, 9.8), layout='constrained')
    for i, (row, rank_label) in enumerate(zip(selected, ('误差较小', '误差居中', '误差最大'), strict=True)):
        for j, (category, key) in enumerate((('fluxes', 'gpp'), ('fluxes', 'npp'),
                                             ('stocks', 'aboveground_biomass_carbon'), ('stocks', 'belowground_biomass_carbon'))):
            ax = axes[i, j]
            for label, color, name in (('teacher', TEACHER, '教师'), ('update14', PARENT, '续训前'), ('update34', CURRENT, '当前')):
                array = values(row, label, category, key)
                if category == 'stocks':
                    array = array - array[0]
                ax.plot(np.arange(1, 31), array, color=color, label=name, lw=1.8 if label != 'update14' else 1.1,
                        linestyle='--' if label == 'update14' else '-')
            entry = row['entry']
            if i == 0:
                ax.set_title(NAMES[key] + ('变化' if category == 'stocks' else ''))
            if j == 0:
                ax.set_ylabel(f'{rank_label}\n{entry["landpoint_id"]} / {entry["year"]}\n' + 'gC/m²/天')
            else:
                ax.set_ylabel('gC/m²/天' if category == 'fluxes' else '相对各自第1日的变化，gC/m²')
            ax.set_xlabel('窗口内第几天')
            ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('连续30天：按当前 GPP 逐日误差选取较小、居中和最大三例\n曲线均为自由连续模拟；生物量列强调增长/下降趋势，起点统一为0', fontsize=14)
    fig.savefig(destination / 'daily_examples.png', dpi=155)
    plt.close(fig)
    # Export every window's curves as a static appendix, not only the three examples.
    fig, axes = plt.subplots(4, 3, figsize=(13, 12), layout='constrained')
    for ax, row in zip(axes.flat, thirty, strict=True):
        for label, color, name in (('teacher', TEACHER, '教师'), ('update14', PARENT, '续训前'), ('update34', CURRENT, '当前')):
            ax.plot(range(1, 31), values(row, label, 'fluxes', 'gpp'), color=color, label=name)
        ax.set(title=f'{row["slice"]} | {row["entry"]["landpoint_id"]} | {row["entry"]["year"]}',
               xlabel='窗口日', ylabel='GPP gC/m²/天')
        ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('全部12个独立30日窗口：GPP逐日曲线', fontsize=15)
    fig.savefig(destination / 'all_30day_gpp.png', dpi=145)
    plt.close(fig)
    (destination / 'physical_metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    render(args.source, args.destination)
