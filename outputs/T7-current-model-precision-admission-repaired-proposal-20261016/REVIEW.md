# 修复版原检查：仅供用户决定

状态：未批准、未部署、未执行；未创建 approval/scope。旧 STOPPED 状态不变，旧授权不可复用。

原 job15866925 在分配后 8 秒、GPU discovery 和模型编译前退出，52 次测量均未开始。它没有给原精度/独立 FD 问题提供数值反证。a833f987 的修复只解除当前入口对旧固定节点的耦合，并在验证前保存真实调度响应；144 项 CPU 回归不是新 GPU 或梯度结论。建议补做完全未开始的原检查，作为调整停线决定的候选，交由用户决定。

固定执行源码：`a833f987ac65e12748f50d4ee9cfd05cbc69d839`。方案 SHA256：`af1cce1067b55941243028d4778380195f003a09d8d40fca481293d6ed10c1b6`。

预定 cwd：`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-current-model-precision-admission-repaired-20261016`。

方案 JSON 保存全部 submission/wrapper/CPU-supervisor argv、新 output/source-cache/approval/diagnostic 路径和固定六步 FD 网格。原科学输入路径保持不变。命令仅供审阅，当前不得执行：

```bash
sbatch --job-name T7-current-repaired --partition gnall --nodes 1 --ntasks 1 --cpus-per-task 4 --gres gpu:1 --time 01:30:00 --no-requeue --chdir /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-current-model-precision-admission-repaired-20261016 --output /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-current-model-precision-admission-repaired-20261016/slurm-%j.stdout.txt --error /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-current-model-precision-admission-repaired-20261016/slurm-%j.stderr.txt --wrap 'exec bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-current-model-precision-admission-repaired-20261016/scripts/hpc/run_shared_current_T7_precision_admission.sh --stage execute --parameters /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-latent-reader-20261004-T7-assembled-input.parameters.pkl --prepared-context /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-expert-real-daily-context-c599236.pkl --training-prepared-manifest /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/training/target-oriented-prepared-1000-a243029/prepared_manifest.json --dataset-manifest /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json --saved-identity-receipt /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.result.b1.optimized-hlo.txt.gz.identity.json --state-input-metadata /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-latent-reader-20261004/T7-state-inputs-20261009.json --turnover-overlay /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-latent-reader-20261004/turnover_leaf_age_fall.guarded-source.py --fp32-reference /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-latent-reader-20261004/T7-B1-state-jvp-completion-20261009.independent-analysis.json --source-cache /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-current-model-precision-admission-repaired-20261016/outputs/T7-current-model-precision-admission-repaired-20261016/source-cache --approval-report /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-current-model-precision-admission-repaired-20261016/approval.v2.json --runtime-preflight /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-current-model-precision-admission-repaired-20261016/runtime-preflight.v2.json --output /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-current-model-precision-admission-repaired-20261016/result.v2.json'
```

冻结版本的 Slurm 脚本仍写死旧 CODE/OUT，因此本方案用 `sbatch --wrap` 直接调用同一版本的通用 wrapper，未修改源码。新 checkout、输出、cache 与 approval 互相一致，不覆盖旧 30d7eb7、approval 或结果。Slurm 默认选节点，允许正常排队；不要求先有空卡。

资源不变：gnall，1 节点/1 task/1 V100/4 实际 Slurm CPU，非独占，90 分钟，24 GiB，最多一次编译/一个作业、不重试。按 CPU ¥0.07/核时与 GPU ¥2.20/卡时，最高估算 `(4×0.07+2.20)×1.5=¥3.72`；实际账单未知。分配开始计时包括启动、测量、保存与释放，保留原 180 秒释放预留；队列等待不计入该执行预算。

原 66,123,713 参数 B1/artifact299/window0/T7 公共 `(flux+stock)/9` loss、原权重/Teacher、完整 3854 状态、4 方向、每方向固定六步与 52 调用（16 native+36 FD）、rtol1e-5/atol1e-6 均不变。state20 严格失败独立保留。434/200 单侧仅作差异观察；2621 是程序坐标偏导，不联动碳库。不得择步追 AD、扩大切图或放宽门槛。

交付应回答：FP64 是否改善当前模型 native 自洽、完整 VJP 重复性及独立 FD 一致性，哪些仍不可判定；从保存 raw 独立复核。一次检查不能保证区分所有舍入、branch 与实现原因；FP64 通过不等于 FP32 B4 通过，也不是训练或提速证据。科学 child exit3 应按保存 JSON 区分完成测量但 state20 未过与运行异常。

本轮实际检查：daily HEAD/branch/clean；ORCJAX 的 jax_orchidee 模块解析仍指向本 daily checkout；JSON 路径及 argv 一致性；本地真实 Bash 词展开和通用 wrapper --dry-run（仅打印命令，未跑 supervisor/model）。复用 a833f987 的 144 项 CPU 回归，未重跑；复用 30d7eb7 的真实 server CPU/Bash 检查仅限未变模型/资产及历史启动行为，不冒充新路径或 --wrap 的服务器检查。

仍未执行：新批准、新不可变 checkout 部署、新路径的实际目标-shell dry-run、fresh approval、sbatch、GPU discovery/preflight/编译/模型测量及释放采样。获得明确新批准后，先部署固定 SHA 并完成目标 shell 参数检查，之后才可按批准范围提交。此次只交付方案，不改写旧停线入口、不恢复授权、不守候。
