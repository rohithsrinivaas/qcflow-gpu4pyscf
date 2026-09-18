import os
import yaml

from pymatgen.core.structure import Molecule

from crn_utils.gpu_pool import map_over_gpus
from crn_utils.file_store import write_result, result_exists

OUTPUT_DIR = os.path.join("outputs", "init_optimized")
TRAJ_DIR = os.path.join("outputs", "trajectories")


def run_one(item):
    """Runs in a worker process pinned to one GPU; must be module-level so it
    can be pickled for the `spawn` multiprocessing context. `pyscf_calc` (which
    pulls in cupy/gpu4pyscf) is imported here, not at module scope, so it never
    touches CUDA before the pool initializer has set CUDA_VISIBLE_DEVICES."""
    from crn_utils import pyscf_calc

    name, molecule, charge, calc_config = item
    trajectory_path = os.path.join(TRAJ_DIR, f"{name}.traj")
    result = pyscf_calc.run_molecule_job(molecule, charge, calc_config,
                                          trajectory_path=trajectory_path)
    write_result(OUTPUT_DIR, name, result)
    return name


def build_work_item(file_path, charge, calc_config):
    """Build the work item for one molecule xyz file, or None if it's already computed."""
    name = os.path.splitext(os.path.basename(file_path))[0]
    if result_exists(OUTPUT_DIR, name):
        return None
    molecule = Molecule.from_file(file_path)
    return (name, molecule, charge, calc_config)


if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    calc_config = config['input_params_step1']
    n_gpus = config.get('gpu', {}).get('n_gpus', 8)

    # Set the xyz files and their charge states
    xyz_files = ["./init_mols/Li.xyz", "./init_mols/PC.xyz", "./init_mols/superoxide.xyz"]
    charge_list = [+1, 0, -1]

    work_items = []
    for file_path, charge in zip(xyz_files, charge_list):
        item = build_work_item(file_path, charge, calc_config)
        if item is not None:
            work_items.append(item)

    print(f"Running {len(work_items)} molecules across {n_gpus} GPUs")
    for result in map_over_gpus(run_one, work_items, n_gpus=n_gpus):
        if isinstance(result, dict) and "error" in result:
            print("FAILED:", result["error"], result["item"])
        else:
            print("Finished", result)
