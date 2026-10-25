"""Shared soil properties at the source litter/soilcarbon call boundaries."""

import jax.numpy as jnp


def bulk_density_source_update(bulk_dens):
    """Apply one source in-place density assignment as a functional value.

    Fortran provenance: ``src_stomate/stomate_litter.f90::littercalc_leak``
    lines1971-1973 and ``stomate_soilcarbon.f90::soilcarbon_leak`` lines1473-1478.
    Each source call multiplies values below500 by1000 (g/cm3 to kg/m3 for
    ordinary raw soil density). The second call reads the first call's value.
    This literal operation is NOT idempotent for arbitrary tiny positive
    inputs. Raw values>=0.5 and already-converted values>=500 are stable after
    the first update. Full cross-step shared writeback for smaller inputs is
    outside the current static-density boundary contract.
    """
    density = jnp.asarray(bulk_dens)
    return jnp.where(density < 500, density * 1000, density)
