import os
from datetime import datetime
import logging
import random
import argparse
import copy
from pprint import pprint

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig()

from typing import Dict, List, Tuple

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN

from monty.serialization import loadfn, dumpfn

from instance_mongodb import instance_mongodb_sei

from ase import units as ase_units

kcal_per_mol_to_eV = 0.0433641

argparser = argparse.ArgumentParser()
argparser.add_argument("--debug", action="store_true", help="Debug mode")
args = argparser.parse_args()


def get_all_graphs(data):
    """Extract all graphs from the data."""
    all_graphs = []
    all_ids = []
    for entry in data:
        all_graphs.append(entry["molecule_graph"])
        all_ids.append(entry["molecule_id"])
    return all_graphs, all_ids


def get_required_schema(
    molecule_graph: MoleculeGraph,
    recombination_data_list: List[Dict],
    molecule_id: str,
    temperature: float = 298.15,
) -> Dict:
    """Convert molecule graph to required schema."""

    # Get the recombination data
    recombination_data, recombination_data_quantities = recombination_data_list

    data = {}
    data["bonds"] = {}
    data["charge"] = molecule_graph.molecule.charge
    data["chemical_system"] = recombination_data["chemsys"]
    data["formula_alphabetical"] = recombination_data["formula_alphabetical"]
    data["molecule_graph"] = molecule_graph.as_dict()
    data["molecule"] = molecule_graph.molecule.as_dict()
    data["molecule_id"] = molecule_id
    data["number_atoms"] = len(molecule_graph.molecule)
    data["number_elements"] = len(molecule_graph.molecule.composition.elements)
    data["partial_charges"] = {}
    data["partial_charges"]["resp"] = recombination_data["output"]["resp"]
    data["partial_charges"]["mulliken"] = recombination_data["output"]["mulliken"]
    data["partial_charges"]["nbo"] = recombination_data_quantities["output"]["nbo"]
    data["partial_spins"] = {}
    data["partial_spins"]["resp"] = None
    data["partial_spins"]["mulliken"] = None
    data["partial_spins"]["nbo"] = None
    data["point_group"] = recombination_data["pointgroup"]
    data["redox"] = {}
    data["redox"]["electron_affinity_eV"] = None
    data["redox"]["ionization_energy_eV"] = None
    data["species"] = recombination_data["calcs_reversed"][0]["species"]
    data["spin_multiplicity"] = molecule_graph.molecule.spin_multiplicity
    data["thermo"] = {}

    # electronic_energy = recombination_data["output"]["final_energy"]
    # electronic_energy_eV = electronic_energy * ase_units.Hartree
    # total_enthalpy = recombination_data["calcs_reversed"][0]["total_enthalpy"]
    # total_enthalpy_eV = total_enthalpy * kcal_per_mol_to_eV
    # zpe = recombination_data["calcs_reversed"][0]["ZPE"]
    # zpe_eV = zpe * kcal_per_mol_to_eV
    # total_entropy = recombination_data["calcs_reversed"][0]["total_entropy"]
    # total_entropy_eV = total_entropy * kcal_per_mol_to_eV / 1000
    # free_energy_eV = (
    #     electronic_energy_eV + total_enthalpy_eV - total_entropy_eV * temperature
    # )

    # data["thermo"]["eV"] = {
    #     "electronic_energy": electronic_energy_eV,
    #     "free_energy": free_energy_eV,
    #     "total_enthalpy": total_enthalpy_eV,
    #     "total_entropy": total_entropy_eV,
    # }
    # data["thermo"]["raw"] = {
    #     "electronic_energy_Ha": electronic_energy,
    #     "rotational_enthalpy_kcal/mol": recombination_data["calcs_reversed"][0][
    #         "rot_enthalpy"
    #     ],
    #     "rotational_entropy_cal/molK": recombination_data["calcs_reversed"][0][
    #         "rot_entropy"
    #     ],
    #     "total_enthalpy_kcal/mol": total_enthalpy,
    #     "total_entropy_cal/molK": total_entropy,
    #     "translational_enthalpy_kcal/mol": recombination_data["calcs_reversed"][0][
    #         "trans_enthalpy"
    #     ],
    #     "translational_entropy_cal/molK": recombination_data["calcs_reversed"][0][
    #         "trans_entropy"
    #     ],
    #     "vibrational_enthalpy_kcal/mol": recombination_data["calcs_reversed"][0][
    #         "vib_enthalpy"
    #     ],
    #     "vibrational_entropy_cal/molK": recombination_data["calcs_reversed"][0][
    #         "vib_entropy"
    #     ],
    # }

    # data["vibration"] = {
    #     "IR_intensities": recombination_data["calcs_reversed"][0]["IR_intens"],
    #     "frequencies": recombination_data["calcs_reversed"][0]["frequencies"],
    #     "mode_vectors": recombination_data["calcs_reversed"][0][
    #         "frequency_mode_vectors"
    #     ],
    # }

    positions = molecule_graph.molecule.cart_coords.tolist()
    data["xyz"] = positions

    return data


def generate_molecule_ids(all_old_ids):
    """Generate new molecule ids."""
    # New molecule ids will be of the form "clibe-<6 digit number>"
    # Generate a random 6 digit number
    new_id = "mvbe-" + str(random.randint(100000, 999999))
    # Check if new_id is in all_old_ids
    if new_id in all_old_ids:
        # If it is, generate a new one
        new_id = generate_molecule_ids(all_old_ids)
    else:
        # If it is not, return it
        return new_id


if __name__ == "__main__":
    """Create the mol_json file containing all the molecules to be used in the model."""

    # Load the current mol_json file
    # current_data = loadfn("data/updated_choli_v3_idsfixed.json")

    current_data = []
    # Store all of the current molecules in a list
   #  list_current_graphs, list_ids = get_all_graphs(current_data)
    list_current_graphs, list_ids = [],[]

    # Get the new structures from the database
    db = instance_mongodb_sei()
    collection = db.tasks

    # Choose the collection
    if args.debug:
        recombination_collection = collection.find(
            {
                "tags.group": "Mg_fragmentation_recombination_quantities",
                "tags.class": "Mg_initial_molecules"
               }
        ).limit(10)
    else:
        recombination_collection = collection.find(
            {
                "tags.group": "Mg_fragmentation_recombination_quantities",
                "tags.class": "Mg_initial_molecules"
            }
        )
    
    print(recombination_collection)
    for idx, recombination_data in enumerate(recombination_collection):
        # Based on if the structure changes or not, write out the fragments
        # to a separate directory
        structure_change = recombination_data["structure_change"]
        # Allow only unique string entries in structure_change
        structure_change = list(set(structure_change))

       # if "unconnected_fragments" in structure_change:
       #     logger.warning(f"Skipping {idx} because it is unconnected")
       #     continue

        tags_recombination = recombination_data["tags"]
        tags_recombination_quantities_ = copy.deepcopy(tags_recombination)
        tags_recombination_quantities_[
            "group"
        ] = "Mg_fragmentation_recombination_quantities"
        tags_recombination_quantities = {"tags.group": "Mg_fragmentation_recombination_quantities", "tags.class": "Mg_initial_molecules"}

        # Find the corresponding recombination_data_quantities
        recombination_quantities_collection = collection.find_one(
            tags_recombination_quantities
        )

        if recombination_quantities_collection == None:
            logger.warning(
                f"Skipping {idx} because it has no quantities, compute single points first."
            )
            continue

        # Get the molecule and molecule graph
        molecule = recombination_data["output"]["initial_molecule"]
        molecule = Molecule.from_dict(molecule)
        molecule_graph = MoleculeGraph.with_local_env_strategy(molecule, OpenBabelNN())

       # Check if the molecule graph is already in the list
        skip_idx = False
        for check_graph in list_current_graphs:
           if molecule_graph.isomorphic_to(check_graph):
               if molecule_graph.molecule.charge == check_graph.molecule.charge:
                   if (
                       molecule_graph.molecule.spin_multiplicity
                       == check_graph.molecule.spin_multiplicity
                   ):
                       logger.warning(
                           f"Skipping {idx} because it is already in the list"
                       )
                       skip_idx = True
                       break
        if skip_idx:
           continue

        # At this point, we have a new molecule graph
        # which is unique and can be added to the list
        molecule_id = generate_molecule_ids(list_ids)
        logger.info(f"Adding {idx} with id {molecule_id}")
        assert molecule_id is not None

        recombination_data_list = [
            recombination_data,
            recombination_quantities_collection,
        ]
        schema_add_ = get_required_schema(
            molecule_graph, recombination_data_list, molecule_id=molecule_id
        )
        logger.info(f"Adding {idx} with id {molecule_id}")

        # Add the new molecule to the list
        current_data.append(schema_add_)

    # Write out the new mol_json file
    # Create a string with the date and time of creating the file
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    # Use `dt_string` to create a new file name
    new_file_name = "mgbe" + dt_string + ".json"

    # Write out the new file
    print(len(current_data))
    dumpfn(current_data, new_file_name)
