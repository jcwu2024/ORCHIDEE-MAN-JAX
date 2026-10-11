"""Independent source declaration and explicit old-reference admission gates."""

import hashlib
from copy import deepcopy

import pytest

from scripts.dev import global_lifecycle_science as science


def fake_source():
    return {
        "science_sha256": science.SCIENCE_SHA256,
        "science_source_pin": science.SCIENCE_PIN,
        "actual_science_source_sha256": "a" * 64,
        "science_source_contract": "exact-declared-new-carbon-core-source-v1",
        "science_source_representations": {"jax_orchidee/fixture.py": "exact Git blob"},
        "expected_code_sha": science.SCIENCE_PIN,
        "claim": "declared scientific changes; strict frozen-output comparison pending",
    }


def fake_identity(strategy="phased_resident_control"):
    lanes = [{"landpoint_id": f"{i:03d}", "inputs_sha256": f"{i:064x}"} for i in range(64)]
    return {
        **fake_source(),
        "contract": science.CONTRACT,
        "process_strategy": science.PROCESS_STRATEGY,
        "stage_name": science.STAGE_NAME,
        "stage_shape": 1024,
        "total_lanes": 2048,
        "candidate_shapes": [1024],
        "unique_real_points": 64,
        "warm_repeats": 5,
        "budget_seconds": 7200,
        "baseline_ref": science.BASELINE_REF,
        "runtime": {
            "python": "fixture",
            "numpy": "fixture",
            "jax": "0.4.38",
            "jaxlib": "0.4.38",
            "x64_enabled": True,
            "device_details": [{"platform": "gpu", "kind": "V100", "id": 0}],
            "backend_platform_version": "fixture CUDA",
        },
        "compiler_environment": {
            "XLA_FLAGS": None,
            "JAX_PLATFORMS": "gpu",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        "gate": science.GATE,
        "precision_policy": science.PRECISION_POLICY,
        "comparison_reference": science.comparison_admission(),
        "lanes": lanes,
        "lane_origins": [
            {
                "lane_index": i,
                "source_index": i % 64,
                "landpoint_id": lanes[i % 64]["landpoint_id"],
                "repeat_index": i // 64,
            }
            for i in range(2048)
        ],
        **science.measurement_fields(strategy),
        "run_binding": {
            "Slurm_job_id": "fixture-job",
            "hostname": "fixture-node",
            "physical_gpu": {"uuid": "fixture-UUID", "total_bytes": 32 * 1024**3},
            "server_resources": {"allocated_CPU_request_count": 4},
            "allocator": "default",
            "preallocate": False,
            "window_start_epoch": 1.0,
        },
    }


@pytest.fixture
def tiny_source(tmp_path, monkeypatch):
    path = "jax_orchidee/fixture.py"
    target = tmp_path / path
    target.parent.mkdir()
    blob = b"value = 1\nother = 2\n"
    target.write_bytes(blob)
    digest = hashlib.sha256(path.encode() + b"\0" + blob + b"\0").hexdigest()
    monkeypatch.setattr(science, "SCIENCE_SHA256", digest)

    def git(root, *args):
        if args[0] in {"ls-tree", "ls-files"}:
            return (path + "\n").encode()
        if args[0] == "show":
            return blob
        if args[0] == "rev-parse":
            return science.SCIENCE_PIN.encode()
        if args[0] == "status":
            assert args[1] == "--porcelain"  # Supported by server Git1.8.3.1.
        return b""

    monkeypatch.setattr(science, "git", git)
    return tmp_path, target, blob


def test_source_accepts_only_exact_blob_or_explicit_uniform_crlf(tiny_source):
    root, target, blob = tiny_source
    lf = science.source_identity(root)
    target.write_bytes(blob.replace(b"\n", b"\r\n"))
    crlf = science.source_identity(root)
    assert crlf["science_sha256"] == lf["science_sha256"]
    assert crlf["actual_science_source_sha256"] != lf["actual_science_source_sha256"]
    target.write_bytes(blob.replace(b"\n", b"\r\n", 1))
    with pytest.raises(ValueError, match="undeclared scientific source bytes"):
        science.source_identity(root)


@pytest.mark.parametrize("kind", ["edited", "untracked", "ignored_extension", "wrong_code"])
def test_undeclared_science_and_wrong_code_fail_closed(tiny_source, kind):
    root, target, blob = tiny_source
    expected = science.SCIENCE_PIN
    if kind == "edited":
        target.write_bytes(blob.replace(b"1", b"3"))
    elif kind in {"untracked", "ignored_extension"}:
        (target.parent / ("extra.py" if kind == "untracked" else "extra.pyd")).write_bytes(b"extra")
    else:
        expected = "0" * 40
    with pytest.raises(ValueError):
        science.source_identity(root, expected_code_sha=expected)


@pytest.mark.parametrize(
    "key,value",
    [
        ("sampling_order", "interleaved_core_wrapper"),
        ("science_sha256", science.OLD_SCIENCE_SHA256),
        ("expected_code_sha", "0" * 40),
        ("measurement_strategy", "legacy_interleaved"),
    ],
)
def test_identity_rejects_wrong_science_order_strategy_and_code(key, value):
    identity = fake_identity()
    identity[key] = value
    with pytest.raises(ValueError):
        science.validate_identity(identity, strategy="phased_resident_control", expected_code_sha=science.SCIENCE_PIN)


def test_pair_common_identity_retains_order_and_excludes_only_two_policy_fields():
    control, release = fake_identity(), fake_identity("phased_release")
    assert science.common_identity(control) == science.common_identity(release)
    wrong = deepcopy(release)
    wrong["sampling_order"] = "interleaved_core_wrapper"
    assert science.common_identity(control) != science.common_identity(wrong)
    with pytest.raises(ValueError, match="sampling_order"):
        science.validate_identity(wrong, strategy="phased_release", expected_code_sha=science.SCIENCE_PIN)
