# qcflow
A customized package for GPU4PySCF calculations &amp; reaction network construction.

This branch replaces the original Q-Chem/FireWorks/MongoDB workflow with **GPU4PySCF**
(GPU-accelerated PySCF) + **Sella** for geometry optimization, run as plain Python
multiprocessing loops across the local GPUs -- no job queue, no database.

* Activate a conda env with `pyscf`, `gpu4pyscf`, `sella`, `ase`, `cupy`, `pymatgen`,
  `molbar==1.1.3` (`pip install molbar==1.1.3` -- pinned because molecule ids
  are derived from its barcode format, which is not guaranteed stable across
  releases), and the Persson-group fork of `mrnet`
  (`pip install git+https://github.com/zhongpc/mrnet.git`) installed.
* Set up your own `config.yaml` (project name, GPU count, DFT/solvent settings).
  A custom SMD solvent can be registered in `crn_utils/pyscf_calc.py`
  (`gpu4pyscf.solvent.smd.solvent_db`) if it isn't one of GPU4PySCF's built-in
  entries.


# The general approach using the qcflow

* Create a directory containing the initial molecules in `init_mols`
* Run `python 1-optimize-init.py` -- this runs directly across the GPUs configured in
  `config.yaml` (`gpu.n_gpus`), writing one result JSON per molecule to
  `outputs/init_optimized/`. Rerunning is safe: molecules that already have a
  result file are skipped.
* Run `python 2-create-mol-json.py` to assign each optimized molecule a
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
