"""Smoke test for a fresh setup: one SMD single point on water with config.yaml's
step-1 settings. Prints the software versions, whether gpu4pyscf's deterministic
mode and cuTENSOR are active, the energy and the wall time.

    python examples/smoke_test.py              # ~20 s on an A100
    python examples/smoke_test.py --repeat 2   # also checks that repeats are bit-identical
    GPU4PYSCF_DETERMINISTIC=1 python examples/smoke_test.py --repeat 2
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
parser.add_argument("--repeat", type=int, default=1)
args = parser.parse_args()

import yaml
from pymatgen.core import Molecule
from crn_utils import pyscf_calc as pc

versions = pc.software_versions()
try:
    from gpu4pyscf.lib import cutensor
    # gpu4pyscf falls back to contract_engine='cupy' when cuTENSOR is unavailable
    versions["cutensor"] = cutensor.cutensor is not None and cutensor.contract_engine is None
except Exception:  # noqa: BLE001
    versions["cutensor"] = False
print("software:", versions)

calc_config = yaml.safe_load(open(args.config))["input_params_step1"]
water = Molecule(["O", "H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.76, 0.59], [0.0, -0.76, 0.59]])

energies = []
for i in range(args.repeat):
    t0 = time.time()
    result = pc.single_point(water, 0, 1, calc_config)
    energies.append(result["energy_Ha"])
    print(f"run {i}: E = {result['energy_Ha']:.12f} Ha  ({time.time() - t0:.0f} s)")

if args.repeat > 1:
    identical = all(e == energies[0] for e in energies)
    print("repeats bit-identical:", identical,
          "" if identical else f"(max spread {max(abs(e - energies[0]) for e in energies):.1e} Ha)")
    if versions["deterministic"] and not identical:
        sys.exit("deterministic mode is on but repeats differ")
print("OK")
