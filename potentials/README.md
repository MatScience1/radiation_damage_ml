# Interatomic Potentials

Place potential files in this directory and set the corresponding paths in `.env`.

---

## ADP — WMoNbZrTiTa (fast classical path)

**File:** `WMoNbZrTiTa.nist.adp.txt`

**Source:** NIST Interatomic Potentials Repository
- URL: https://www.ctcms.nist.gov/potentials/
- Search for: W-Mo-Nb-Zr-Ti-Ta ADP

**Species order in potential file header:**
```
W Mo Nb Zr Ti Ta
```
This order is hard-coded in `CFG.eval.adp_species_order` in `pipeline/config.py`.
If you use a different ADP file with a different element ordering, update that list.

**Set in `.env`:**
```
ADP_POTENTIAL=/absolute/path/to/WMoNbZrTiTa.nist.adp.txt
```

---

## MLIP — MACE-MP-0 (accurate path, auto-download)

MACE-MP-0 downloads automatically when `MLIP_BACKEND=mace` (the default).
The checkpoint is cached in `~/.cache/mace/` on first run.

To pin a specific checkpoint file:
```
MLIP_MODEL=/path/to/mace_mp_medium.model
```

---

## MLIP — GRACE (accurate path, manual download)

**Source:** https://github.com/ICAMS/grace-tensorpotential

Install the package:
```bash
pip install git+https://github.com/ICAMS/grace-tensorpotential.git
```

Download a checkpoint (contact ICAMS/RUB or check their releases).

**Set in `.env`:**
```
MLIP_BACKEND=grace
MLIP_MODEL=/path/to/grace_model.pth
```

---

## MLIP — SevenNet-0 (accurate path, auto-download)

```bash
pip install sevenn
```

SevenNet-0 downloads automatically when `MLIP_BACKEND=sevennet` and
`MLIP_MODEL` is empty (uses the built-in `7net-0` universal model ID).

**Set in `.env`:**
```
MLIP_BACKEND=sevennet
# MLIP_MODEL=  # leave empty for auto-download, or point to a .pt checkpoint
```

---

## Switching backends at runtime

```bash
# Use MACE for all complex defects (default)
MLIP_BACKEND=mace python run_pipeline.py

# Use GRACE with a specific checkpoint
MLIP_BACKEND=grace MLIP_MODEL=potentials/grace_model.pth python run_pipeline.py

# Use SevenNet
MLIP_BACKEND=sevennet python run_pipeline.py
```

The routing threshold (number of vacancies above which MLIP is used instead
of ADP) is set by `CFG.eval.mlip_complexity_threshold` (default: 2).
