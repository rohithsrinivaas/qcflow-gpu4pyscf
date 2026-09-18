"""Perform single points storing the NBO data."""
import os
import logging
import yaml
import sys
import argparse

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN

from fireworks import LaunchPad, Workflow
from fireworks.user_objects.dupefinders.dupefinder_exact import DupeFinderExact

from atomate.qchem.fireworks.core import SinglePointFW
from atomate.common.powerups import add_tags

from instance_mongodb import instance_mongodb_sei

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

    # Launch the calculation
    lp = LaunchPad.from_file("/global/homes/n/nessa/config/my_launchpad.yaml")

    # Get the command line arguments
    args = get_cli()
    logger.info("Dry run: {}".format(args.dryrun))

    # Get the new structures from the database
    db = instance_mongodb_sei()
    collection = db.tasks

    # This name will be passed onto to {'tags.class'}
    CLASSNAME = "Mg_fragmentation_recombination_quantities" 
    # refers to the single point after frequency calculation

    # Parameters to import
    with open(os.path.join("/global/homes/n/nessa/config", "libe_parameters.yaml"), "r") as f:
        params = yaml.safe_load(f)
    nbo_params = {"nbo_params": {"version": 7}}
    params.update(nbo_params)

    recombination_collection = collection.find(
        {
            "tags.group": "Mg_fragmentation_recombination",
            "tags.class": "Mg_initial_molecules"
        }
    )

    # This collection stores the output of the
    # single point calculation, check to see if it already exists
    calculated_collection = collection.find({"tags.group": CLASSNAME, "tags.class": "Mg_initial_molecules"})
    all_graphs = get_all_graphs(calculated_collection)
    logger.info(f"Found {len(all_graphs)} graphs in the database")

    count_structures = 0

    for idx, recombination_data in enumerate(recombination_collection):

        # Based on if the structure changes or not, write out the fragments
        # to a separate directory
        structure_change = recombination_data["structure_change"]
        # Allow only unique string entries in structure_change
        structure_change = list(set(structure_change))

        tags = recombination_data["tags"]
        tags["group"] = CLASSNAME
        tags["class"] = "Mg_initial_molecules"

        # Get the molecule graph
        if "optimized_molecule" not in recombination_data["output"]:
            continue
        molecule = recombination_data["output"]["optimized_molecule"]
        molecule = Molecule.from_dict(molecule)
        molecule_graph = MoleculeGraph.with_local_env_strategy(molecule, OpenBabelNN())

        # Check if the molecule graph is already in the list
        if check_already_completed(molecule_graph, all_graphs):
            logger.warning(f"Skipping {idx} because it is already in the list")
            continue

        calc_name = "Mg_fragmentation_recombination_quantities"

        count_structures += 1

        if not args.dryrun:

            # Run the simplest workflow available
            firew = SinglePointFW(
                 molecule=molecule,
                qchem_input_params=params,
                name=calc_name,
                db_file=">>db_file<<",
              #  spec={"_dupefinder": DupeFinderExact()},
            )

            # Create a workflow for just one simple firework
            wf = Workflow([firew], name=calc_name)

            # Label the set appropriately
            wf = add_tags(wf, tags)

            # Add set to launchpad
            lp.add_wf(wf)

    logger.info(f"Added {count_structures} structures to the launchpad")
