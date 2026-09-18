"""Benchmark reuse_dm=True vs reuse_dm=False on PC optimization.

Patches GPU4PySCFCalculator.calculate() to record SCF cycle count per step,
then runs the full optimization twice and prints a timing + iteration summary.
"""
import time
import sys
import os
import yaml
import numpy as np
from pymatgen.core import Molecule

# ── patch calculate() to record cycle counts before importing gpu4pyscf ────────
_cycle_log = []   # filled by the patched calculate()

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)
calc_config_base = config["input_params_step1"]

MOLECULE_FILE = "init_mols/PC.xyz"
CHARGE = -1   # PC anion: 55 electrons (odd) → UKS


def run_benchmark(reuse_dm: bool):
    from crn_utils import pyscf_calc
    from crn_utils.pyscf_calc import GPU4PySCFCalculator

    # Monkey-patch calculate to log cycle counts
    original_calculate = GPU4PySCFCalculator.calculate

    cycles_this_run = []

    def patched_calculate(self, atoms=None, properties=("energy", "forces"),
                           system_changes=None):
        original_calculate(self, atoms, properties,
                            system_changes or pyscf_calc.all_changes)
        cycles_this_run.append(getattr(self.mf, "cycles", None))

    GPU4PySCFCalculator.calculate = patched_calculate

    mol = Molecule.from_file(MOLECULE_FILE)
    calc_config = {**calc_config_base, "reuse_dm": reuse_dm}

    t0 = time.perf_counter()
    result = pyscf_calc.run_molecule_job(mol, charge=CHARGE, calc_config=calc_config,
                                          trajectory_path=None)
    elapsed = time.perf_counter() - t0

    # Restore original
    GPU4PySCFCalculator.calculate = original_calculate

    return elapsed, cycles_this_run, result


def main():
    label = {True: "reuse_dm=True ", False: "reuse_dm=False"}

    results = {}
    for flag in [False, True]:          # run without first so no warm-up bias
        print(f"\n{'='*60}")
        print(f"  Running {label[flag]}")
        print(f"{'='*60}")
        elapsed, cycles, result = run_benchmark(flag)
        results[flag] = (elapsed, cycles, result)
        print(f"  Converged : {result.get('converged')}")
        print(f"  Energy    : {result.get('energy_Ha'):.6f} Ha")
        print(f"  Opt steps : {len(cycles)}")
        print(f"  Wall time : {elapsed:.1f} s")
        if cycles:
            valid = [c for c in cycles if c is not None]
            print(f"  SCF iters : {valid}  (mean {np.mean(valid):.1f})")

    print(f"\n{'='*60}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*60}")
    for flag in [False, True]:
        elapsed, cycles, _ = results[flag]
        valid = [c for c in cycles if c is not None]
        mean_iters = np.mean(valid) if valid else float("nan")
        print(f"  {label[flag]}  {elapsed:6.1f} s  "
              f"steps={len(cycles)}  mean_SCF_iters={mean_iters:.1f}")

    t_no, t_yes = results[False][0], results[True][0]
    speedup = t_no / t_yes if t_yes > 0 else float("nan")
    print(f"\n  Speedup (reuse_dm=True vs False): {speedup:.2f}x")


if __name__ == "__main__":
    main()
