import sys
import os
import yaml
import logging
import argparse

from atomate.qchem.fireworks.core import FrequencyFlatteningOptimizeFW
from atomate.qchem.fireworks.core import SinglePointFW
from atomate.common.powerups import add_tags

from fireworks import LaunchPad, Workflow

from instance_mongodb import instance_mongodb_sei

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN

from fireworks.user_objects.dupefinders.dupefinder_exact import DupeFinderExact

# Launch the calculation
lp = LaunchPad.from_file("/global/homes/n/nessa/config/my_launchpad.yaml")
# lp = LaunchPad.from_file("/home/svijay/fw_config/my_launchpad.yaml")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig()


def get_cli():
    """Get the command line arguments."""
    args = argparse.ArgumentParser()
    args.add_argument(
        "--dryrun", action="store_true", help="Do not launch the workflow."
    )
    return args.parse_args()


def get_all_graphs(collection):
    """Extract all graphs from the data."""
    all_graphs = []
    for entry in collection:
        molecule = entry["input"]["initial_molecule"]
        molecule = Molecule.from_dict(molecule)
        # Create the molecule graph
        molecule_graph = MoleculeGraph.with_local_env_strategy(molecule, OpenBabelNN())
        all_graphs.append(molecule_graph)
    return all_graphs


def check_already_completed(molecule_graph, all_graphs):
    """Check if the molecule graph is already completed."""
    for graph in all_graphs:
        if molecule_graph.isomorphic_to(graph):
            if molecule_graph.molecule.charge == graph.molecule.charge:
                if (
                    molecule_graph.molecule.spin_multiplicity
                    == graph.molecule.spin_multiplicity
                ):
                    return True
    return False


if __name__ == "__main__":

    # Get the command line arguments
    args = get_cli()
    logger.info("Dry run: {}".format(args.dryrun))

    # Parameters to import
    print(os.path.join("/global/homes/n/nessa/config", "libe_parameters.yaml"))
    with open(os.path.join("/global/homes/n/nessa/config", "libe_parameters.yaml"), "r") as f:
        params = yaml.safe_load(f)

    # Get the database and create a collection to store the structures in
    db = instance_mongodb_sei()
    collection = db.tasks
    collection_initial_structures = db.Mg_cluster_initial_structures_12_06_2023

    # Get the fragments of the molecules
    recombination_collection = collection_initial_structures.find(
        {
            "tags.type_structure": "fragment",
            "tags.group": "initial_structures_fragmentation_recombination",
        }
    )

    # Get older structures that have already been calculated.
    calculated_collection = collection.find(
        {
            "tags.type_structure": "fragment",
            "tags.group": "Mg_fragmentation_recombination",
        }
    )
    all_graphs = get_all_graphs(calculated_collection)
    print(len(all_graphs))

    NAME = "recombination_calculation"

    output_dir = os.path.join("outputs", "new_recombination_structures")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count_structures = 0
    accepted_structures = 0

    # Write out the recombination structures all in separate xyz files.
    for idx, recombination_data in enumerate(recombination_collection):
        tags_recombination = recombination_data.pop("tags")
        tags_recombination["group"] = "Mg_fragmentation_recombination"

        molecule_graph = MoleculeGraph.from_dict(recombination_data)

        # Increment the count
        count_structures += 1

        # Check if the molecule graph is already in the list
        if check_already_completed(molecule_graph, all_graphs):
            logger.warning(f"Skipping {idx} because it is already in the list")
            continue
        molecule = molecule_graph.molecule
        accepted_structures += 1

        if args.dryrun:
            # Write out the molecule
            molecule.to(fmt="xyz", filename=os.path.join(output_dir, f"{idx}.xyz"))

        elif len(molecule) == 1:
            firew = SinglePointFW(
                molecule=molecule,
                qchem_input_params=params,
                name="single_point",
                db_file=">>db_file<<",
            )
            # Create a workflow for just one simple firework
            wf = Workflow([firew], name=NAME)

            tags_recombination["group"] = "Mg_fragmentation_recombination_quantities"

            # Label the set appropriately
            wf = add_tags(wf, tags_recombination)

            # Add set to launchpad
            lp.add_wf(wf)

        else:
            firew = FrequencyFlatteningOptimizeFW(
                molecule=molecule,
                qchem_input_params=params,
                db_file=">>db_file<<",
                #spec={"_dupefinder": DupeFinderExact()},
            )

            

            # Create a workflow for just one simple firework


            # wf = Workflow([firew, firw_2], name=NAME)
            # firw_2 = SinglePointFW()
            
            wf = Workflow([firew], name=NAME)

            # Label the set appropriately
            wf = add_tags(wf, tags_recombination)

            # Add set to launchpad
            lp.add_wf(wf)

    logger.info(f"{count_structures} structures processed for calculation.")
    logger.info(f"{accepted_structures} structures accepted for calculation.")
