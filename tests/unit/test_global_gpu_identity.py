"""GPU identity metadata only; no JAX, CUDA context or model execution."""
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.hpc import global_gpu_identity as identity


def xml(minor=5, uuid="GPU-five", pci="00000000:3D:00.0"):
    return f"""<nvidia_smi_log><driver_version>525.85.12</driver_version>
      <gpu id="{pci}"><product_name>Tesla V100-SXM2-32GB</product_name>
      <uuid>{uuid}</uuid><minor_number>{minor}</minor_number>
      <pci><pci_bus_id>{pci}</pci_bus_id></pci></gpu></nvidia_smi_log>"""


def proc(minor=5, uuid="GPU-five", pci="0000:3d:00.0"):
    return [{"minor": minor, "uuid": uuid, "pci": pci, "path": "driver_record"}]


def test_remapped_visible_zero_resolves_exact_global_minor_and_UUID():
    result = identity.resolve("5", "0", xml(), proc())
    assert result["uuid"] == "GPU-five" and result["device_minor"] == 5
    assert result["CUDA_VISIBLE_DEVICES_preserved"] == "0"
    assert result["slurm_global_gpu_id"] == "5" and result["model_execution"] is False
    assert result["CUDA_selector_UUID_resolution_verified"] is False


def test_UUID_selector_also_requires_exact_driver_record():
    assert identity.resolve("GPU-five", "GPU-five", xml(), proc())["uuid"] == "GPU-five"


@pytest.mark.parametrize("global_id,visible,inventory,driver,reason", [
    ("7", "0", xml(), proc(), "unknown/ambiguous"),
    ("5", "0", xml(uuid="GPU-other"), proc(), "not accessible"),
    ("5", "0", xml(minor=0), proc(), "minor/PCI"),
    ("5", "0", xml(pci="0000:1a:00.0"), proc(), "minor/PCI"),
    ("0,5", "0", xml(), proc(), "one unmodified"),
    ("5", "0,1", xml(), proc(), "one preserved"),
])
def test_unknown_and_inconsistent_mapping_never_selects_first_visible(global_id, visible, inventory, driver, reason):
    with pytest.raises(ValueError, match=reason):
        identity.resolve(global_id, visible, inventory, driver)


def test_proc_metadata_parse_matches_observed_driver_format(tmp_path):
    path = tmp_path / "pci-driver-record"
    path.mkdir()
    (path / "information").write_text("Model:\tTesla V100\nGPU UUID:\tGPU-five\nDevice Minor:\t5\nBus Location:\t0000:3d:00.0\n")
    result = identity.proc_inventory(tmp_path)
    assert result[0]["minor"] == 5 and result[0]["uuid"] == "GPU-five"


def test_empty_NVML_failure_has_explicit_exit_and_saved_diagnostics(monkeypatch, tmp_path, capsys):
    destination = tmp_path / "identity.json"
    monkeypatch.setattr(identity.sys, "argv", ["helper", "--slurm-global-id", "5", "--cuda-visible-devices", "0", "--output", str(destination)])
    monkeypatch.setattr(identity.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=6, stdout="", stderr=""))
    assert identity.main() == 2
    assert "exit=6" in capsys.readouterr().err
    assert '"status": "failed_metadata_only"' in destination.read_text()


def test_selected_card_optional_failure_records_exit_stdout_stderr_without_stopping(tmp_path):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Bash unavailable")
    source = (Path(__file__).resolve().parents[2] / "scripts/hpc/slurm_global_server_gpu_b64.sh").read_text()
    function = source.split("selected_gpu_diagnostic() {", 1)[1].split("\n}", 1)[0]
    command = """set -Eeuo pipefail
RUN_ROOT=$PWD
GPU_UUID=GPU-five
nvidia-smi() {
  printf '%s\\n' "$*"
  printf 'simulated optional NVML failure\\n' >&2
  return 6
}
selected_gpu_diagnostic() {""" + function + "\n}\nselected_gpu_diagnostic selected-gpu-processes --query-compute-apps=gpu_uuid,pid --format=csv\nprintf 'startup_continues\\n'\n"
    result = subprocess.run([bash, "-s"], input=command, text=True, capture_output=True, cwd=tmp_path)
    assert result.returncode == 0 and "startup_continues" in result.stdout
    assert (tmp_path / "gpu-diagnostic-exits.tsv").read_text().strip() == "selected-gpu-processes\t6"
    assert "-i GPU-five" in (tmp_path / "selected-gpu-processes.stdout.txt").read_text()
    assert "simulated optional NVML failure" in (tmp_path / "selected-gpu-processes.stderr.txt").read_text()
