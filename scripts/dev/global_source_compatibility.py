"""Exact-byte certificate for the audited DIFFUCO host-cache repair only."""
import hashlib
import subprocess

DIFFUCO_PATH = "jax_orchidee/sechiba/diffuco.py"
OLD_SHA256 = "04fc1193e85d2deca0f97cc06008989053ccaf956db3dc9552dd6b9cc2549d22"
NEW_SHA256 = "42073dd14f427db8ed78803c90cd5a64b677bf822d1edf6b6fbed7aa97166a14"
BASELINE_SHA256 = "11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5"
WINDOWS_BASELINE_SHA256 = "4151db252c62cf24899300c7cb2eac783405888fce1384030332d4622aa2c196"
OLD_BLOCK = b'''@lru_cache(maxsize=32)
def _diffuco_pft_column_mask(nvm: int, pft_index: int):
    return jnp.arange(int(nvm)) == int(pft_index)
'''
NEW_BLOCK = b'''@lru_cache(maxsize=32)
def _diffuco_pft_column_mask(nvm: int, pft_index: int):
    # This host cache can be first populated inside a JAX trace. Caching a
    # jnp.arange result would retain that trace's DynamicJaxprTracer and lift
    # it into later programs as an undeclared AOT input. Cache only the exact
    # immutable host Boolean mask; JAX converts it in the consuming trace.
    mask = np.arange(int(nvm)) == int(pft_index)
    mask.setflags(write=False)
    return mask
'''


def audit_diffuco_bytes(data):
    """Return certified baseline bytes; reject any unapproved module change."""
    actual = hashlib.sha256(data).hexdigest()
    if actual == OLD_SHA256:
        return data, False
    if actual != NEW_SHA256 or data.count(NEW_BLOCK) != 1:
        raise ValueError("unapproved DIFFUCO execution source bytes")
    baseline = data.replace(NEW_BLOCK, OLD_BLOCK, 1)
    if hashlib.sha256(baseline).hexdigest() != OLD_SHA256:
        raise ValueError("DIFFUCO certificate transformation differs outside approved helper")
    return baseline, True


def source_identity(root):
    paths = subprocess.check_output(["git", "ls-files", "jax_orchidee"], cwd=root, text=True).splitlines()
    actual, baseline = hashlib.sha256(), hashlib.sha256()
    repaired = False
    for path in paths:
        data = (root / path).read_bytes()
        certified = data
        if path == DIFFUCO_PATH:
            certified, repaired = audit_diffuco_bytes(data)
        actual.update(path.encode() + b"\0" + data + b"\0")
        baseline.update(path.encode() + b"\0" + certified + b"\0")
    representations = {BASELINE_SHA256: "canonical Linux LF source bytes",
                       WINDOWS_BASELINE_SHA256: "exact audited Windows checkout source bytes"}
    if baseline.hexdigest() not in representations:
        raise ValueError("science source changed outside certified execution-only cache repair")
    certificate = {"active": repaired, "contract": "exact-DIFFUCO-host-mask-cache-repair-v1",
                   "source_path": DIFFUCO_PATH, "original_file_sha256": OLD_SHA256,
                   "actual_file_sha256": NEW_SHA256 if repaired else OLD_SHA256,
                   "baseline_full_source_sha256": BASELINE_SHA256,
                   "baseline_checkout_bytes_sha256": baseline.hexdigest(),
                   "baseline_checkout_representation": representations[baseline.hexdigest()],
                   "actual_full_source_sha256": actual.hexdigest(),
                   "compatibility_scope": "exact helper-only immutable host Boolean mask; all other source bytes unchanged",
                   "claim": "audited execution-only equivalence; source bytes differ" if repaired else "original source bytes"}
    return {"science_sha256": BASELINE_SHA256, "actual_science_source_sha256": actual.hexdigest(),
            "execution_source_compatibility": certificate}
