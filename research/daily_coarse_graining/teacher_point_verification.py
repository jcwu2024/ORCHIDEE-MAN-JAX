"""Real production-reader and absence-of-loose proof, without model execution."""
from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from .teacher_assets import (MAP_NAME, asset_read_text, close_asset_readers,
    fingerprint, loose_sha256, safe_path)
from .teacher_point_storage import FIELDS, Migration, atomic_json, file_stat


def tree_equal(left, right, location="root"):
    if dataclasses.is_dataclass(left):
        if type(left) is not type(right):raise ValueError("class drift " + location)
        for f in dataclasses.fields(left):tree_equal(getattr(left,f.name),getattr(right,f.name),location+"."+f.name)
    elif isinstance(left,dict):
        if left.keys() != right.keys():raise ValueError("mapping drift " + location)
        for key in left:tree_equal(left[key],right[key],location+"."+str(key))
    elif isinstance(left,(tuple,list)):
        if type(left) is not type(right) or len(left) != len(right):raise ValueError("sequence drift " + location)
        for i,(a,b) in enumerate(zip(left,right)):tree_equal(a,b,location+"."+str(i))
    elif hasattr(left,"shape") and hasattr(left,"dtype"):
        a,b = np.asarray(left),np.asarray(right)
        if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
            raise ValueError("array bytes/dtype/shape drift " + location)
    elif left != right:raise ValueError("value drift " + location)


def _cleanup_owned_view(view: Path, expected_files: dict[str,tuple]):
    """Exact known proof files only; refuse namespace/identity changes."""
    actual = {p.relative_to(view).as_posix():p for p in view.rglob("*") if p.is_file()}
    if set(actual) != set(expected_files):raise ValueError("proof cleanup namespace drift")
    for name,path in actual.items():
        safe_path(path)
        if fingerprint(path) != expected_files[name]:raise ValueError("proof cleanup identity drift")
    for path in actual.values():
        if os.name == "nt":os.chmod(path,0o600)
        path.unlink()
    for path in sorted((p for p in view.rglob("*") if p.is_dir()),key=lambda p:len(p.parts),reverse=True):path.rmdir()
    view.rmdir()


def verify_release_noncode(release) -> dict:
    """Keep every non-Python source gate in historical-output inspection."""
    from .teacher_data_release import _resolve
    raw = release.raw
    identities = {"teacher_config":raw["runtime"]["config"],
                  "pft_catalog":raw["runtime"]["pft_catalog"],
                  "typed_contract":raw["contracts"]["typed_supplement"]}
    result = {}
    for name,identity in identities.items():
        path = safe_path(_resolve(identity["path"]))
        before = fingerprint(path)
        digest = loose_sha256(path)
        if digest != identity["sha256"] or fingerprint(path) != before:
            raise ValueError("release non-code hash drift: "+name)
        result[name] = {"path":str(path),"sha256":digest}
    return result


def prove_point(migration: Migration, point: str, *, scan_full_index=False) -> dict:
    """Restore once, compare every annual reader, then exercise actual absence.

    The shadow release keeps the exact original full scientific manifest. Its
    full indices use the existing post-acceptance no-file-scan mode; bounded
    samples use the selected point's original reference classes. No source file
    is renamed/removed to create this absence proof.
    """
    from . import daily_markov_contract as dm, markov_dataset as md, typed_sidecar as typed, teacher_shards as teacher
    from . import teacher_assets, teacher_point_storage
    from .daily_operator.training_data import DailyProcessTrainingAdapter, coherent_process_training_examples
    started = time.monotonic()
    record = migration.record(point)
    if record is None or record["stage"] not in {"published","reader_verified"}:
        raise ValueError("published complete point required for proof")
    view = safe_path(migration.work / (point + ".proof-view"))
    restore_start = time.monotonic()
    restoration = migration.restore(point,view)
    restoration_seconds = time.monotonic()-restore_start
    root = migration.root
    contract = dm.daily_markov_contract_from_metadata(migration.manifest["markov_contract"])
    typed_contract = typed.load_typed_sidecar_contract(typed.COHERENT_CONTRACT_PATH)
    annual = []; reader_start = time.monotonic()
    # This reads accepted historical outputs. The current source checkout is
    # deliberately different from the frozen producer; do not relabel it as
    # the historical source. Release self-hash/contracts remain enforced and
    # _completed_metadata below verifies each record's actual input identities.
    plan = teacher.load_plan(Path(migration.manifest["plan"]),verify_release_sources=False)
    noncode = verify_release_noncode(plan.data_release)
    if plan.plan_sha256 != migration.manifest["plan_sha256"]:
        raise ValueError("original generation-plan identity drift")
    entries = {(r.landpoint_id,r.year):r for r in plan.entries}
    for r in record["records"]:
        a = dm.load_markov_shard(root/r["shard"],contract=contract)
        b = dm.load_markov_shard(view/r["shard"],contract=contract)
        ta = typed.read_typed_sidecar_shard(root/r["typed_shard"],contract=typed_contract,expected_days=a.days)
        tb = typed.read_typed_sidecar_shard(view/r["typed_shard"],contract=typed_contract,expected_days=b.days)
        tree_equal(a,b);tree_equal(ta,tb)
        if not np.array_equal(a.day_index,ta[2]):raise ValueError("typed/base day alignment")
        for horizon in (1,7,30):
            for start in sorted({0,min(123,a.days-horizon),a.days-horizon}):
                tree_equal(a.window(start,horizon),b.window(start,horizon))
        metadata = json.loads(asset_read_text(root/r["metadata"]))
        restored_metadata = json.loads((view/r["metadata"]).read_bytes())
        tree_equal(metadata,restored_metadata)
        worker = "/".join(r["checkpoint"].split("/")[:2])
        tree_equal(teacher._load_checkpoint(root/worker,metadata),teacher._load_checkpoint(view/worker,restored_metadata))
        completed = teacher._completed_metadata(plan,root/worker,entries[(point,r["year"])],
            plan_sha256=plan.plan_sha256,git_head=migration.manifest["teacher_git_head"],
            preceding_checkpoint_sha256=r["preceding_checkpoint_sha256"])
        if completed != metadata:raise ValueError("production completion validator rejected annual record")
        annual.append({"year":r["year"],"days":a.days,"spatial_split":r["spatial_split"],"temporal_split":r["temporal_split"],
                       "base_typed_masks_day_bits_classes_equal":True,"T1_T7_T30_equal":True,
                       "checkpoint_state_equal":True,"completed_metadata_pass":True})
        del a,b,ta,tb
        gc.collect()
    annual_seconds = time.monotonic()-reader_start
    # Exact owned restored files are removed after their successful full restore
    # comparison. Source originals above are never this cleanup base.
    for name,row in record["rows"].items():
        path = safe_path(view/name)
        if loose_sha256(path) != row["sha256"] or any(file_stat(path)[k] != row[k] for k in ("size","mode","mtime_ns")):
            raise ValueError("restored view drift before absence proof")
        if os.name == "nt":os.chmod(path,0o600)
        path.unlink()
    (view/"dataset_manifest.json").write_bytes((root/"dataset_manifest.json").read_bytes())
    if loose_sha256(view/"dataset_manifest.json") != migration.manifest_sha256:raise ValueError("proof manifest bytes changed")
    (view/"points").mkdir()
    pack = root/"points"/(point+".sqlite");view_pack = view/"points"/pack.name
    shutil.copyfile(pack,view_pack);os.chmod(view_pack,0o444)
    if loose_sha256(view_pack) != record["container"]["sha256"]:raise ValueError("proof pack copy hash drift")
    source_map = json.loads((root/MAP_NAME).read_bytes())
    source_map["points"] = {point:source_map["points"][point]}
    atomic_json(view/MAP_NAME,source_map)
    expected = {p.relative_to(view).as_posix():fingerprint(p) for p in view.rglob("*") if p.is_file()}
    if any((view/name).exists() for name in record["rows"]):raise ValueError("loose proof members remain")
    absence_start = time.monotonic()
    # The actual full index constructors preserve all original scientific records.
    parent = md.load_dataset_index(view/"dataset_manifest.json",verify_hashes=False,verify_files=False)
    coherent = typed.load_coherent_typed_sidecar_index(view/"dataset_manifest.json",verify_hashes=False,verify_files=False)
    if len(parent.shards) != len(migration.manifest["shards"]) or len(coherent.shards) != len(parent.shards):
        raise ValueError("full scientific index inventory drift")
    point_parent = dataclasses.replace(parent,shards=tuple(r for r in parent.shards if r.landpoint_id == point))
    point_typed = dataclasses.replace(coherent,parent_index=point_parent,
        shards=tuple(r for r in coherent.shards if r.parent.landpoint_id == point))
    adapter = DailyProcessTrainingAdapter.from_contract(contract)
    probes = []
    for split in sorted({r["temporal_split"] for r in record["records"]}):
        sample = next(point_parent.samples(temporal_split=split))
        windows = [next(point_parent.windows(h,temporal_split=split)) for h in (1,7,30)]
        typed_sample = next(point_typed.samples(temporal_split=split))
        example = next(coherent_process_training_examples(point_typed,adapter,temporal_split=split))
        if sample["sample_landpoint_id"] != point or len(windows) != 3:raise ValueError("point sample identity drift")
        probes.append({"temporal_split":split,"sample_class":type(sample).__name__,"typed_sample_class":type(typed_sample).__name__,
                       "training_example_class":type(example).__module__+"."+type(example).__name__,"windows":[1,7,30]})
    # Checkpoint and metadata readers also use physically absent logical members.
    for r in record["records"]:
        meta = json.loads(asset_read_text(view/r["metadata"]))
        worker = "/".join(r["checkpoint"].split("/")[:2])
        teacher._load_checkpoint(view/worker,meta)
    absence_seconds = time.monotonic()-absence_start
    close_asset_readers()
    _cleanup_owned_view(view,expected)
    scan = None
    if scan_full_index:
        scan_start = time.monotonic()
        actual = typed.load_coherent_typed_sidecar_index(root/"dataset_manifest.json",verify_hashes=False)
        if len(actual.shards) != len(parent.shards):raise ValueError("actual root full index count drift")
        scan = {"records":len(actual.shards),"verify_files":True,"verify_hashes":False,
                "seconds":time.monotonic()-scan_start,"unselected_payloads_not_hashed":True}
    migration.assert_release()
    for name,row in record["rows"].items():
        if file_stat(safe_path(root/name)) != {k:row[k] for k in file_stat(root/name)}:
            raise ValueError("original source stat drift after proof")
    sources = {m.__name__:{"path":m.__file__,"sha256":loose_sha256(Path(m.__file__))}
               for m in (dm,md,typed,teacher,teacher_assets,teacher_point_storage,sys.modules[__name__])}
    checkout = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=checkout,text=True).strip()
    source_paths = [str(Path(v["path"]).relative_to(checkout)) for v in sources.values()]
    subprocess.run(["git","diff","--exit-code","HEAD","--",*source_paths],cwd=checkout,check=True,capture_output=True)
    return {"status":"PRODUCTION_POINT_READERS_AND_ABSENT_LOOSE_PASS","landpoint_id":point,
            "container_sha256":record["container"]["sha256"],"snapshot_sha256":record["snapshot_sha256"],
            "dataset_manifest_sha256":migration.manifest_sha256,"source_modules":sources,"git_commit":commit,
            "historical_producer_source_revalidated":False,"release_self_hash_and_contracts_verified":True,
            "generation_plan_sha256":plan.plan_sha256,"actual_annual_input_hashes_verified":True,
            "release_noncode_sources":noncode,
            "annual_records":annual,"all_200_original_objects_byte_verified":True,
            "restoration":restoration,"restore_seconds":restoration_seconds,"annual_comparison_seconds":annual_seconds,
            "absence_probes":probes,"absence_seconds":absence_seconds,
            "absence_full_index_records":len(parent.shards),"absence_full_index_verify_files":False,
            "absence_full_index_verify_hashes":False,"selected_absent_assets_verified_on_open":True,
            "selected_index_references_original_classes":True,"actual_root_full_file_scan":scan,
            "proof_view_removed":not view.exists(),"source_files_removed":0,
            "source_stat_unchanged":True,"model_transition_calls":0,"seconds":time.monotonic()-started}
