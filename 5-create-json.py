import logging
import os

import yaml
from ase import units as ase_units
from monty.serialization import dumpfn
from mrnet.core.mol_entry import MoleculeEntry
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.core.structure import Molecule

from crn_utils.file_store import iter_results
from crn_utils.mrnet_utils import remove_high_energy_mol_entries

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig()

HARTREE2EV = ase_units.Hartree


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


print("LOADING")
mol_docs = [
    doc for doc in iter_results(os.path.join("outputs", "frag_recombine_optimized"))
    if "error" not in doc and doc.get("converged")
]
print("Total number of mol_docs (converged, no errors):", len(mol_docs))


print("CREATING MOL ENTRIES")
mol_entries = []
for doc in mol_docs:
    initial_molecule = Molecule.from_dict(doc["initial_molecule"])
    mol_graph = MoleculeGraph.from_local_env_strategy(initial_molecule, OpenBabelNN())

    mol_entries.append(MoleculeEntry(
        molecule=initial_molecule,
        energy=doc["energy_Ha"],
        entry_id=doc["_id"],
        mol_graph=mol_graph,
    ))

print("Total number of mol_entries:", len(mol_entries))

print("FILTERING -- removing high energy molecules")
mol_entries_filtered = remove_high_energy_mol_entries(mol_entries)
good_ids = {entry.entry_id for entry in mol_entries_filtered}
mol_docs = [doc for doc in mol_docs if doc["_id"] in good_ids]
print("Total number of mol_entries_filtered:", len(mol_entries_filtered))
print("Total number of mol_docs (after cleaning):", len(mol_docs))


print("BUILDING")
entries = []
for doc in mol_docs:
    mol = Molecule.from_dict(doc["molecule"])  # optimized geometry

    entry = {}
    entry["molecule_id"] = doc["_id"]
    entry["molecule"] = doc["molecule"]
    entry["charge"] = mol.charge
    entry["spin_multiplicity"] = mol.spin_multiplicity
    entry["species"] = [str(s) for s in mol.species]
    entry["xyz"] = mol.cart_coords.tolist()
    entry["number_atoms"] = len(mol)
    entry["number_elements"] = len(mol.composition.elements)
    entry["composition"] = mol.composition.get_el_amt_dict()
    entry["elements"] = list(entry["composition"].keys())
    entry["formula_alphabetical"] = mol.composition.alphabetical_formula
    entry["chemical_system"] = mol.composition.chemical_system

    entry["spin_contamination"] = doc["spin_contamination"]

    entry["point_group"] = doc["point_group"]
    entry["rotational_symmetry_number"] = doc["rotational_symmetry_number"]
    entry["basis"] = doc["basis"]
    entry["xc"] = doc["xc"]
    entry["smd_solvent"] = doc["smd_solvent"]

    entries.append(entry)


########### save the entries for QChem run #####################
save_filename = "entries_" + config["workflow_tags"]["group"] + "_qchem.json"
dumpfn(entries, save_filename)

print(f"WRITING {len(entries)} ENTRIES to {save_filename}")
print("COMPLETED")
