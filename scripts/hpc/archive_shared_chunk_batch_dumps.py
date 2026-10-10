"""Verify a bounded archive before removing completed compiler-dump duplicates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

NAMES=("dumps-runtime-preflight-default-20261001-vmap",
       "dumps-chunk-vmap-default-20261001-vmap",
       "dumps-chunk-vmap-default-20261001-vmap2",
       "dumps-stage-screen-default-20261001-vmap",
       "dumps-shape-preserved-default-20261001-vmap")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--overlay",type=Path,required=True)
    p.add_argument("--remove-verified-loose",action="store_true")
    args=p.parse_args()
    allowed=Path("/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer").resolve()
    overlay=args.overlay.resolve()
    if not overlay.is_relative_to(allowed) or overlay.name!="shared-command-buffer-loops-20261001":
        raise ValueError("unexpected task overlay")
    roots=[overlay/name for name in NAMES]
    for root in roots:
        if not root.is_dir() or root.is_symlink() or root.resolve().parent!=overlay:
            raise ValueError("missing or unsafe closed dump directory")
    archives=overlay/"archives"
    archives.mkdir(exist_ok=True)
    archive=archives/"shared-chunk-batch-completed-dumps-20261001.tar.gz"
    index=archive.with_suffix(".index.json")
    if archive.exists() or index.exists():
        raise FileExistsError("preserve previous archive/index")
    expected={}
    counts=[]
    for root in roots:
        files=directories=byte_count=0
        for path in (root,*sorted(root.rglob("*"))):
            if path.is_symlink() or not path.resolve().is_relative_to(overlay):
                raise ValueError("unsafe dump entry")
            name=path.relative_to(overlay).as_posix()
            if path.is_file():
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                expected[name]=digest
                files+=1
                byte_count+=path.stat().st_size
            elif path.is_dir():
                expected[name]=None
                directories+=1
            else:
                raise ValueError("non-file/directory dump entry")
        counts.append({"directory":str(root),"files":files,"directories":directories,"bytes":byte_count})
    with tarfile.open(archive,"w:gz") as target:
        for root in roots:
            target.add(root,arcname=root.name)
    observed={}
    with tarfile.open(archive,"r:gz") as saved:
        for member in saved:
            name=member.name.rstrip("/")
            if name in observed or name not in expected or not (member.isfile() or member.isdir()):
                raise ValueError("archive has unexpected/duplicate/unsupported member")
            if member.isfile():
                stream=saved.extractfile(member)
                observed[name]=hashlib.sha256(stream.read()).hexdigest()
            else:
                observed[name]=None
    if observed!=expected:
        raise ValueError("archive member/content fidelity failed")
    # Closed inputs must still match after the complete archive readback.
    for name,digest in expected.items():
        path=overlay/name
        if digest is not None and hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError("dump changed during archive verification")
    report={"schema":"shared_chunk_batch_dump_archive_v1","status":"verified",
        "archive":str(archive),"archive_bytes":archive.stat().st_size,
        "archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
        "members":len(expected),"directories":counts,"verification":"every archive member type/name and complete regular-file SHA256 equals the closed input; full tar stream readback",
        "restore_command":f"tar -xzf {archive} -C {overlay}","removed_verified_loose":False}
    index.write_text(json.dumps(report,indent=2)+"\n")
    if args.remove_verified_loose:
        for root in roots:
            if root.resolve().parent!=overlay or not root.resolve().is_relative_to(allowed):
                raise ValueError("recursive removal escaped declared overlay")
            shutil.rmtree(root)
        report["removed_verified_loose"]=True
        report["status"]="verified_and_compacted"
        index.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__=="__main__":
    main()
