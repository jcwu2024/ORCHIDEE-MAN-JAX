import hashlib, json, subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[4]
path='jax_orchidee/sechiba/diffuco.py'
old=subprocess.check_output(['git','show','8b7732aaa0c75d58f1aa515a7fccbadd628f353b:'+path],cwd=root)
new=(root/path).read_bytes()
start=old.index(b'@lru_cache(maxsize=32)\ndef _diffuco_pft_column_mask')
end=old.index(b'\n\ndef _diffuco_insert_pft_column(',start)
ns=new.index(b'@lru_cache(maxsize=32)\ndef _diffuco_pft_column_mask')
ne=new.index(b'\n\ndef _diffuco_insert_pft_column(',ns)
assert old[:start]==new[:ns] and old[end:]==new[ne:]
digest=hashlib.sha256()
for name in subprocess.check_output(['git','ls-files','jax_orchidee'],cwd=root,text=True).splitlines():
    data=old if name==path else (root/name).read_bytes()
    digest.update(name.encode()+b'\0'+data+b'\0')
print(json.dumps({'old_sha':hashlib.sha256(old).hexdigest(),'new_sha':hashlib.sha256(new).hexdigest(),
                  'baseline_sha':digest.hexdigest(),'old_block':old[start:end].decode(),'new_block':new[ns:ne].decode()},indent=2))
