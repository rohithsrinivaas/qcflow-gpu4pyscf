# qcflow
A customized package for GPU4PySCF calculations &amp; reaction network construction.

This branch (`gpu4pyscf-1.8`) replaces the original Q-Chem/FireWorks/MongoDB workflow
with **GPU4PySCF** (GPU-accelerated PySCF) + **Sella** for geometry optimization, run as
plain Python multiprocessing loops across the local GPUs -- no job queue, no database.
It targets GPU4PySCF 1.8 (the group's fork, below) and adds the SCF-guess reuse, coarser
production grids and SOSCF fallback described in `crn_utils/pyscf_calc.py`.

## Setup

1. **Python environment** (Perlmutter, A100, CUDA 12):
   ```
   conda env create -f environment.yaml && conda activate qcflow
   mkdir -p $CONDA_PREFIX/etc/conda/activate.d && cp env/cutensor-activate.sh $CONDA_PREFIX/etc/conda/activate.d/
   conda activate qcflow   # again, so the hook runs
   ```
   The hook puts cuTENSOR on `LD_LIBRARY_PATH`; without it GPU4PySCF silently falls back
   to slower contractions.
2. **GPU4PySCF, the group's fork** (`github.com/oliverhvidsten/gpu4pyscf`, tag `v1.8.1-det.1`).
   It is upstream GPU4PySCF 1.8 plus a deterministic mode and two loader fixes; the version
   reports as `1.8.1+det`. Either install the prebuilt A100 wheel
   ```
   pip install --no-deps /global/cfs/cdirs/matgen/shared/gpu4pyscf-det/gpu4pyscf_cuda12x-1.8.1+det-*.whl
   ```
   or build it from source (about 30 min): clone the tag and follow the build snippet in
   the fork's `DETERMINISM.md`.
   The stock `gpu4pyscf-cuda12x` wheel from PyPI also works with this code, without the
   deterministic mode.
3. **Smoke test**: from the repository root,
   `python examples/smoke_test.py --repeat 2`
   prints the versions, whether cuTENSOR is active (it should be; if not, the activation
   hook from step 1 is missing), and a water SMD energy twice. With `GPU4PYSCF_DETERMINISTIC=1` the two energies must be bit-identical.
4. **Configure** `config.yaml`: project name, GPU count, DFT/solvent settings. Every
   SCF/grid/optimizer key is documented there and in the `pyscf_calc.py` docstring. The
   custom SMD solvent `EC:EMC` is registered in `crn_utils/pyscf_calc.py`; add others the
   same way.

## Deterministic (bit-reproducible) runs

Stock GPU4PySCF gives slightly different energies and forces on every run (1e-13 Ha,
1e-7 eV/Å) because its GPU kernels sum in a scheduling-dependent order; geometry
optimizations can then end in different minima. The fork's deterministic mode removes
this at no measurable cost. Turn it on with `gpu: {deterministic: true}` in
`config.yaml` (the workers then run with `GPU4PYSCF_DETERMINISTIC=1`), or export that
variable yourself for ad-hoc scripts. Each result records the mode under `"software"`.
Validated for H, C, N, O, F, P, S; heavier elements are untested -- see the fork's
`DETERMINISM.md`, "Limitations".

## The general approach using the qcflow

* Create a directory containing the initial molecules in `init_mols`
* Run `python 1-optimize-init.py` -- this runs directly across the GPUs configured in
  `config.yaml` (`gpu.n_gpus`), writing one result JSON per molecule to
  `outputs/init_optimized/`. Rerunning is safe: molecules that already have a
  result file are skipped.
* Run `python 2-create-init-json.py` to assign each optimized molecule a
  deterministic id from its MolBar structural barcode (topology +
  stereochemistry + charge) and dedup on that barcode, into
  `mol_fromFlattening_step2.json`
* Run `python 3-fragment-recombine.py` to get the fragmentation/recombination
  candidates, stored as flat JSON files under `outputs/initial_graphs/`
* Run `python 4-compute_recombine.py` to run the accepted recombination
  candidates across the GPUs, writing to `outputs/step4_optimized/`
* Run `python 5-create-json.py` to build the final CRN entries
  (`entries_<group>_step5.json`) from the step 4 results
* Apply HiPGen [nix-shell] to analyze the results based on the json file

`crn_utils/calculate_*.py` are the original FireWorks/Q-Chem utilities and are not used
by this workflow.
