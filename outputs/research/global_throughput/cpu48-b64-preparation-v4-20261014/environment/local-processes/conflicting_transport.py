import os,sys,subprocess
os.environ.update({'OMP_NUM_THREADS': '1', 'OMP_THREAD_LIMIT': '1', 'MKL_NUM_THREADS': '64', 'OPENBLAS_NUM_THREADS': '64', 'NUMEXPR_NUM_THREADS': '64', 'XLA_FLAGS': '--xla_force_host_platform_device_count=48 --xla_cpu_multi_thread_eigen=false', 'JAX_PLATFORMS': 'cuda', 'JAX_ENABLE_X64': 'false', 'JAX_DISABLE_JIT': 'true', 'JAX_NUM_CPU_DEVICES': '48', 'TF_NUM_INTRAOP_THREADS': '1'})
command=sys.argv[1:]
if os.name=='nt': raise SystemExit(subprocess.call(command,env=dict(os.environ)))
os.execvpe(command[0],command,dict(os.environ))
