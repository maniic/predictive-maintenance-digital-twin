# The dataset

## What C-MAPSS is

C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) is NASA's
benchmark for engine prognostics: simulated turbofan engines run from healthy
operation to failure, published by the NASA Prognostics Center of Excellence for
the 2008 PHM data challenge.

Each engine starts with an unknown amount of initial wear, develops a fault at
some point, and degrades until failure. Training trajectories run all the way to
failure; test trajectories are truncated some time before it, and a separate file
gives the true remaining life at that cut-off point.

| Dataset | Train engines | Test engines | Operating conditions | Fault modes |
|---------|--------------|--------------|---------------------|-------------|
| FD001 | 100 | 100 | 1 (sea level) | HPC degradation |
| FD002 | 260 | 259 | 6 | HPC degradation |
| FD003 | 100 | 100 | 1 (sea level) | HPC + fan degradation |
| FD004 | 249 | 248 | 6 | HPC + fan degradation |

(NASA's own `readme.txt` lists FD004 as 248 train / 249 test. The files
themselves contain 249 train and 248 test engines, which is what the loader and
these tables report.)

## Columns

26 space-separated columns per row, no header:

| Column | Meaning |
|---|---|
| 1 | engine id |
| 2 | cycle number (one cycle ≈ one flight) |
| 3–5 | operational settings 1–3 (altitude, Mach number, throttle resolver angle) |
| 6–26 | sensor measurements 1–21 |

The 21 sensors cover temperatures at the fan inlet, LPC outlet, HPC outlet and
LPT outlet; pressures at the fan inlet, bypass duct and HPC outlet; physical and
corrected fan and core speeds; engine pressure ratio; fuel flow ratio; bypass
ratio; burner fuel-air ratio; bleed enthalpy; demanded fan speeds; and HPT/LPT
coolant bleed. Names and units are in `config/config.yaml`, and the original
NASA description is preserved verbatim at `data/raw/readme.txt`.

Several sensors are constant within a sub-dataset (sensor 1 sits at 518.67
throughout FD001). The preprocessor drops only sensors with *exactly* zero
variance — see [architecture.md](architecture.md#preprocessing).

## Why it is committed to this repository

The 13 data files are tracked in git. That is a deliberate choice:

- **It is redistributable.** C-MAPSS is a work of the United States Government,
  published as open data by NASA with no redistribution restriction.
- **It is small in the clone.** The working tree is 43 MB, but the data is highly
  compressible ASCII: it accounts for about 11.5 MB of a roughly 13 MB clone.
  Removing it would require rewriting history and would save around 11 MB, in
  exchange for a network dependency on every clone.
- **It makes the project reproducible in three commands.** Clone, install, run.
  Nothing to download, no dead link to chase in two years.

## Verifying it

Every file is byte-identical to NASA's official distribution. The checksums are
recorded in `data/raw/CHECKSUMS.sha256` and can be verified at any time:

```bash
python scripts/fetch_data.py --verify
```

```
All 13 C-MAPSS files present and verified against NASA checksums.
```

CI runs this check on every push, so a corrupted or silently altered data file
fails the build.

## Re-fetching it

If `data/raw/` is ever emptied, or you want the data in a checkout that excludes
it:

```bash
python scripts/fetch_data.py           # download what is missing, then verify
python scripts/fetch_data.py --force   # re-download and overwrite
```

The script downloads from NASA's public mirror, checks the SHA-256 of every file
against the committed manifest **before** writing anything, and refuses to write
if the upstream archive has changed:

```
https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
```

The archive is nested — the outer zip contains `CMAPSSData.zip`, which contains
the text files plus NASA's *Damage Propagation Modeling* paper. The paper is not
committed here; it is worth reading and the download above includes it.

## Citation

> A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for
> Aircraft Engine Run-to-Failure Simulation", in *Proceedings of the 1st
> International Conference on Prognostics and Health Management (PHM08)*,
> Denver, CO, October 2008.
