# External Data

Large forcing and static input files are intentionally not stored in Git.
Set `ORCHIDEE_DATA_ROOT` to the directory containing the layout described in
`manifests/paper_pft14_data.yaml`.

The historical local development layout (`data/forcing`, `data/INPUTDIR_ZZ`,
and `data/MICT_BIOE/Input`) remains a supported fallback when the environment
variable is not set.
