# Historical Fortran Run Protocol

`paper_250919/` is an immutable copy of the paper-era launch protocol used to
derive the PFT14 configuration. It is retained as scientific provenance, not
as a portable or supported launcher.

The files intentionally preserve historical absolute cluster paths, including
the original account name. Those paths are not credentials and must not be
executed or edited in place. Portable JAX runs use `configs/`, the
`ORCHIDEE_*_ROOT` environment variables, and `orchidee-jax` instead.
