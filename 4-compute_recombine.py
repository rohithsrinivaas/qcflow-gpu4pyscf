import argparse
import json
import os
import logging
import yaml
from pymatgen.analysis.graphs import MoleculeGraph

from crn_utils.fragmentation_recombination import FragmentReconnect
from crn_utils.file_store import JsonFileCollection, iter_results, write_result
from crn_utils.utils import (
    get_all_graphs_from_collection,
    build_dedup_index,
    is_duplicate,
    get_external_duplicate_ids,
)
from crn_utils.gpu_pool import map_over_gpus

OUTPUT_DIR = os.path.join("outputs", "frag_recombine_optimized")
TRAJ_DIR = os.path.join("outputs", "trajectories")
INITIAL_GRAPHS_DIR = os.path.join("outputs", "initial_graphs")


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig()


def run_one(item):
    """Runs in a worker process pinned to one GPU; must be module-level so it
    can be pickled for the `spawn` multiprocessing context."""
    from crn_utils import pyscf_calc

    doc_id, molecule, charge, tags, calc_config = item
    trajectory_path = os.path.join(TRAJ_DIR, f"{doc_id}.traj")
    result = pyscf_calc.run_molecule_job(molecule, charge, calc_config,
                                          trajectory_path=trajectory_path)
    result["_id"] = doc_id
    result["tags"] = tags
    write_result(OUTPUT_DIR, doc_id, result)
    return doc_id


def build_work_items(calc_config, group_name):
    initial_graphs_collection = JsonFileCollection(INITIAL_GRAPHS_DIR)

    # Structures already computed in a previous run of this step (skip errored
    # entries -- they have no "initial_molecule" field yet and should be retried)
    already_computed = [
        result for result in iter_results(OUTPUT_DIR) if "error" not in result
    ]
    all_graphs = get_all_graphs_from_collection(already_computed)
    logger.info(f"{len(all_graphs)} structures already computed")
    already_computed_index = build_dedup_index(all_graphs)

    recombination_candidates = initial_graphs_collection.find(
        {"tags.group": FragmentReconnect.groupname}
    )

    work_items = []
    count_structures = 0
    count_skipped_external = 0
    count_skipped_internal = 0
    for entry in recombination_candidates:
        count_structures += 1
        entry = dict(entry)
        tags = entry.pop("tags")
        doc_id = entry.pop("_id")
        tags["group"] = group_name + "_fragmentation_recombination"

        if doc_id in external_duplicate_ids:
            count_skipped_external += 1
            logger.warning(f"Skipping {doc_id} because it matches an external structure")
            continue

        if doc_id in internal_duplicate_ids:
            count_skipped_internal += 1
            logger.warning(f"Skipping {doc_id} because it duplicates another PC candidate")
            continue

        molecule_graph = MoleculeGraph.from_dict(entry)
        if is_duplicate(molecule_graph, already_computed_index):
            logger.warning(f"Skipping {doc_id} because it is already computed")
            continue

        molecule = molecule_graph.molecule
        logger.info(f"Accepted molecule: {molecule.formula}")
        work_items.append((doc_id, molecule, molecule.charge, tags, calc_config))

    logger.info(f"{count_structures} structures processed for calculation.")
    logger.info(f"{len(work_items)} structures accepted for calculation.")
    return work_items


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=int, default=0,
                         help="Start index into the deduplicated work-item list "
                              "(for splitting across parallel SLURM jobs/nodes).")
    parser.add_argument("--limit", type=int, default=None,
                         help="Number of work items this invocation should process, "
                              "starting at --offset. Default: all remaining.")
    parser.add_argument("--n_gpus", type=int, default=None,
                         help="Override config.yaml's gpu.n_gpus (e.g. 4 for one "
                              "Perlmutter GPU node).")
    parser.add_argument("--count_only", action="store_true",
                         help="Print the total deduplicated work-item count and exit "
                              "without running any calculations.")
    args = parser.parse_args()

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    calc_config = config['input_params_step4']
    n_gpus = args.n_gpus if args.n_gpus is not None else config.get('gpu', {}).get('n_gpus', 8)

    work_items = build_work_items(calc_config, config["workflow_tags"]["group"])

    if args.count_only:
        print(f"TOTAL_WORK_ITEMS: {len(work_items)}")
        raise SystemExit(0)

    end = None if args.limit is None else args.offset + args.limit
    shard = work_items[args.offset:end]
    logger.info(f"Processing shard [{args.offset}:{end}] -> {len(shard)} of {len(work_items)} work items")

    for result in map_over_gpus(run_one, shard, n_gpus=n_gpus):
        if isinstance(result, dict) and "error" in result:
            print("FAILED:", result["error"], result["item"])
        else:
            print("Finished", result)
