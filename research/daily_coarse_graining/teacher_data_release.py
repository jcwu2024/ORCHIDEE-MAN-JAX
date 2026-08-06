"""Versioned identity for a coherent Teacher training-data release."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from research.daily_coarse_graining.typed_sidecar import (
    COHERENT_CONTRACT_PATH,
    TypedSidecarContract,
    load_typed_sidecar_contract,
)

ROOT = Path(__file__).resolve().parents[2]
LEGACY_SCHEMA_VERSION = "canonical_teacher_data_release_v1"
SCHEMA_VERSION = "canonical_teacher_data_release_v2"
DEFAULT_CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"
DEFAULT_PFT_CATALOG = ROOT / "configs" / "pft_catalogs" / "orchidee_man_paper_250919.json"
DEFAULT_PRODUCER_SOURCES = (
    ROOT / "research" / "daily_coarse_graining" / "daily_markov_contract.py",
    ROOT / "research" / "daily_coarse_graining" / "supervised_learnability_pilot.py",
    ROOT / "research" / "daily_coarse_graining" / "teacher_data_release.py",
    ROOT / "research" / "daily_coarse_graining" / "teacher_production.py",
    ROOT / "research" / "daily_coarse_graining" / "teacher_shards.py",
    ROOT / "research" / "daily_coarse_graining" / "typed_capture_masks.py",
    ROOT / "research" / "daily_coarse_graining" / "typed_sidecar.py",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def python_source_sha256(path: str | Path) -> str:
    """Hash UTF-8 Python source after portable newline normalization."""

    text = Path(path).read_text(encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_sha256(value: Mapping[str, Any], *, omit: str | None = None) -> str:
    payload = dict(value)
    if omit is not None:
        payload.pop(omit, None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def python_tree_sha256(path: str | Path) -> str:
    """Hash Python source contents and portable paths, excluding runtime caches."""

    root = Path(path).resolve()
    inventory = {
        source.relative_to(root).as_posix(): python_source_sha256(source) for source in sorted(root.rglob("*.py"))
    }
    if not inventory:
        raise ValueError(f"Teacher source tree contains no Python files: {root}")
    return canonical_sha256(inventory)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _portable(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


@dataclass(frozen=True)
class TeacherDataRelease:
    path: Path
    sha256: str
    release_id: str
    dataset_id: str
    teacher_source_sha256: str
    teacher_config: Path
    markov_contract_sha256: str
    typed_contract: TypedSidecarContract
    raw: Mapping[str, Any]


def freeze_teacher_data_release(
    output: str | Path,
    *,
    release_id: str,
    teacher_config: str | Path = DEFAULT_CONFIG,
    pft_catalog: str | Path = DEFAULT_PFT_CATALOG,
    typed_contract: str | Path = COHERENT_CONTRACT_PATH,
    teacher_source: str | Path = ROOT / "jax_orchidee",
    producer_sources: tuple[str | Path, ...] = DEFAULT_PRODUCER_SOURCES,
) -> Path:
    """Freeze the exact code and contracts used by one coherent data release."""

    if not release_id or any(character.isspace() for character in release_id):
        raise ValueError("release_id must be a non-empty whitespace-free identifier")
    teacher_source = _resolve(teacher_source)
    teacher_config = _resolve(teacher_config)
    pft_catalog = _resolve(pft_catalog)
    contract = load_typed_sidecar_contract(_resolve(typed_contract))
    if contract.generation_policy != "single_pass_continuous_teacher":
        raise ValueError("canonical release requires the coherent typed contract")
    sources = tuple(_resolve(path) for path in producer_sources)
    required = (teacher_source, teacher_config, pft_catalog, contract.path, *sources)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing Teacher release inputs: " + ", ".join(missing))

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_pre_production",
        "release_id": release_id,
        "dataset_id": contract.parent_dataset_id,
        "teacher_source": {
            "path": _portable(teacher_source),
            "python_tree_sha256": python_tree_sha256(teacher_source),
            "hash_mode": "utf8_lf",
        },
        "runtime": {
            "config": {
                "path": _portable(teacher_config),
                "sha256": sha256_file(teacher_config),
            },
            "pft_catalog": {
                "path": _portable(pft_catalog),
                "sha256": sha256_file(pft_catalog),
            },
        },
        "contracts": {
            "markov_contract_sha256": contract.parent_contract_sha256,
            "typed_supplement": {
                "path": _portable(contract.path),
                "sha256": sha256_file(contract.path),
                "contract_sha256": contract.sha256,
            },
        },
        "producer_source_files": [
            {
                "path": _portable(source),
                "sha256": python_source_sha256(source),
                "hash_mode": "utf8_lf",
            }
            for source in sources
        ],
        "generation": {
            "transition_policy": "single_pass_continuous_teacher",
            "artifacts": ["base_shard", "typed_shard", "year_end_checkpoint"],
            "dataset_manifest_schema": contract.dataset_manifest_schema,
        },
    }
    payload["release_sha256"] = canonical_sha256(payload)
    result = _atomic_json(Path(output), payload)
    load_teacher_data_release(result)
    return result


def load_teacher_data_release(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> TeacherDataRelease:
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") not in {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION}:
        raise ValueError("unsupported Teacher data-release schema")
    actual_release_hash = canonical_sha256(raw, omit="release_sha256")
    if raw.get("release_sha256") != actual_release_hash:
        raise ValueError("Teacher data-release self-hash mismatch")
    if raw.get("status") != "frozen_pre_production":
        raise ValueError("Teacher data release is not frozen for production")

    teacher_source = raw.get("teacher_source", {})
    teacher_root = _resolve(teacher_source.get("path", ""))
    expected_teacher_hash = str(teacher_source.get("python_tree_sha256", ""))
    if teacher_source.get("hash_mode") != "utf8_lf":
        raise ValueError("Teacher source hash mode is not portable")
    runtime = raw.get("runtime", {})
    teacher_config = _resolve(runtime.get("config", {}).get("path", ""))
    pft_catalog = _resolve(runtime.get("pft_catalog", {}).get("path", ""))
    typed_identity = raw.get("contracts", {}).get("typed_supplement", {})
    typed_path = _resolve(typed_identity.get("path", ""))
    typed_contract = load_typed_sidecar_contract(typed_path)

    if verify_sources:
        if python_tree_sha256(teacher_root) != expected_teacher_hash:
            raise ValueError("Teacher source-tree hash drift")
        for name, source, identity in (
            ("Teacher config", teacher_config, runtime.get("config", {})),
            ("PFT catalog", pft_catalog, runtime.get("pft_catalog", {})),
            ("typed supplement contract", typed_path, typed_identity),
        ):
            if not source.is_file() or sha256_file(source) != identity.get("sha256"):
                raise ValueError(f"{name} hash drift")
        for identity in raw.get("producer_source_files", ()):
            source = _resolve(identity.get("path", ""))
            if identity.get("hash_mode") != "utf8_lf":
                raise ValueError(f"data producer source hash mode drift: {source}")
            if not source.is_file() or python_source_sha256(source) != identity.get("sha256"):
                raise ValueError(f"data producer source hash drift: {source}")

    if typed_contract.sha256 != typed_identity.get("contract_sha256"):
        raise ValueError("typed supplement semantic contract hash drift")
    if typed_contract.generation_policy != "single_pass_continuous_teacher":
        raise ValueError("Teacher data release uses a historical sidecar contract")
    dataset_id = str(raw.get("dataset_id", ""))
    if typed_contract.parent_dataset_id != dataset_id:
        raise ValueError("typed supplement dataset family mismatch")
    markov_hash = str(raw.get("contracts", {}).get("markov_contract_sha256", ""))
    if typed_contract.parent_contract_sha256 != markov_hash:
        raise ValueError("typed supplement Markov contract mismatch")
    if raw.get("generation", {}).get("transition_policy") != "single_pass_continuous_teacher":
        raise ValueError("Teacher data release does not require single-pass generation")
    declared_dataset_schema = raw.get("generation", {}).get(
        "dataset_manifest_schema"
    )
    if declared_dataset_schema is not None and (
        declared_dataset_schema != typed_contract.dataset_manifest_schema
    ):
        raise ValueError("Teacher data release dataset-manifest schema drift")
    if raw.get("schema_version") == SCHEMA_VERSION and declared_dataset_schema is None:
        raise ValueError("Teacher data release does not bind its dataset schema")

    return TeacherDataRelease(
        path=path,
        sha256=actual_release_hash,
        release_id=str(raw["release_id"]),
        dataset_id=dataset_id,
        teacher_source_sha256=expected_teacher_hash,
        teacher_config=teacher_config,
        markov_contract_sha256=markov_hash,
        typed_contract=typed_contract,
        raw=raw,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--release-id", required=True)
    freeze.add_argument("--teacher-config", type=Path, default=DEFAULT_CONFIG)
    freeze.add_argument("--pft-catalog", type=Path, default=DEFAULT_PFT_CATALOG)
    freeze.add_argument("--typed-contract", type=Path, default=COHERENT_CONTRACT_PATH)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--release", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        path = freeze_teacher_data_release(
            args.output,
            release_id=args.release_id,
            teacher_config=args.teacher_config,
            pft_catalog=args.pft_catalog,
            typed_contract=args.typed_contract,
        )
        release = load_teacher_data_release(path)
    else:
        release = load_teacher_data_release(args.release)
    print(
        json.dumps(
            {
                "release": str(release.path),
                "release_id": release.release_id,
                "release_sha256": release.sha256,
                "teacher_source_sha256": release.teacher_source_sha256,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
