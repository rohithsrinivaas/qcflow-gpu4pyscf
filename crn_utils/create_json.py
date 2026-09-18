from json import loads
from typing import List
import copy
from pprint import pprint
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig()

import numpy as np
import re

from monty.serialization import dumpfn, loadfn

from pymongo import UpdateOne
from pymongo.errors import OperationFailure

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from pymatgen.analysis.local_env import OpenBabelNN

from atomate.qchem.database import QChemCalcDb

from mrnet.src.mrnet.core.mol_entry import MoleculeEntry, MoleculeEntryError
from mrnet.src.mrnet.core.reactions import bucket_mol_entries

from qrrho import QuasiRRHO
from rrho_shifted import ShiftedRRHO

from instance_mongodb import instance_mongodb_sei
from ase import units as ase_units

kcal_per_mol_to_eV = 0.0433641


def remove_high_energy_mol_entries(
    mol_entries: List[MoleculeEntry],
) -> List[MoleculeEntry]:
    """
    For molecules of the same isomorphism and charge, remove the ones with higher free
    energies.

    Args:
        mol_entries: a list of molecule entries

    Returns:
        low_energy_entries: molecule entries with high free energy ones removed
    """

    # convert list of entries to nested dicts
    buckets = bucket_mol_entries(mol_entries, keys=["formula", "num_bonds", "charge"])
    # print(mol_entries)

    all_entries = []
    for formula in buckets:
        for num_bonds in buckets[formula]:
            for charge in buckets[formula][num_bonds]:

                # filter mols having the same formula, number bonds, and charge
                low_energy_entries = []
                for entry in buckets[formula][num_bonds][charge]:

                    # try to find an entry_i with the same isomorphism to entry
                    idx = -1
                    for i, entry_i in enumerate(low_energy_entries):
                        if entry.mol_graph.isomorphic_to(entry_i.mol_graph) and entry.molecule.spin_multiplicity == entry_i.molecule.spin_multiplicity:
                            idx = i
                            break

                    if idx >= 0:
                        # entry has the same isomorphism as entry_i
                        if entry.get_free_energy() is not None and low_energy_entries[idx].get_free_energy() is not None:
                            if (
                                entry.get_free_energy()
                                < low_energy_entries[idx].get_free_energy()
                            ):
                                low_energy_entries[idx] = entry

                    else:
                        # entry with a unique isomorphism
                        low_energy_entries.append(entry)

                all_entries.extend(low_energy_entries)

    return all_entries


if __name__ == "__main__":
    db = instance_mongodb_sei()

    entries = list()

    try:
        db["Mg_DME_fragmentation_recombination"].create_index("molecule_id", unique=True)
        # db["Mg_fragmentation_recombination"].create_index("molecule_id", unique=True)
    except OperationFailure:
        print("Index already existed")

    print("LOADING")

    mol_docs = list(db["Mg_DME_fragmentation_recombination"].find())
    # mol_docs = list(db["Mg_fragmentation_recombination"].find())
    # mol_quant = list(db["Mg_fragmentation_recombination_quantities"].find())
    print(len(mol_docs))
    # print(len(mol_quant))
    mol_quantities = []
    mol_documents = []

    for idx, mol_doc in enumerate(mol_docs):
        # Find the corresponding recombination_data_quantities
        tags_moldoc = mol_doc["tags"]
        tags_mol_quant_ = copy.deepcopy(tags_moldoc)
        tags_mol_quant_[
            "group"
        ] = "Mg_DME_fragmentation_recombination_quantities"
        # tags_mol_quant_["group"] = ["fragmentation_recombination_quantities"]
        # tags_mol_quant_[
        #     "group"
        # ] = "Mg_fragmentation_recombination_quantities"
        smiles_tag = mol_doc["smiles"]
        formula_pretty_tag = mol_doc["formula_pretty"]
        formula_alph_tag = mol_doc["formula_alphabetical"]
        # basis_tag = mol_doc["calcs_reversed"][0]["input"]["rem"]["basis"]
        # method_tag = mol_doc["calcs_reversed"][0]["input"]["rem"]["method"]
        optimized_molecule_tag = mol_doc["output"]["optimized_molecule"]
        # task_label = optimized_data["task_label"]
        # molecule_name = task_label[:-4] + "_" + basis_tag + "_" + method_tag

        tags_mol_quantities = {"tags": tags_mol_quant_, 
                    # "smiles":smiles_tag, "formula_pretty":formula_pretty_tag,
                    # "formula_alphabetical":formula_alph_tag,
                    # # "calcs_reversed.0.input.rem.basis":basis_tag,
                    # # "calcs_reversed.0.input.rem.method":method_tag,
                    # "input.initial_molecule":optimized_molecule_tag,
                }
    
        quantities_collection = db["Mg_DME_fragmentation_recombination_quantities"].find_one(
            tags_mol_quantities
        )
        # quantities_collection = db["Mg_fragmentation_recombination_quantities"].find_one(
        #     tags_mol_quantities
        # )

        if quantities_collection is not None:
            mol_quantities.append(quantities_collection)
            mol_documents.append(mol_doc)
        else:
            logger.warning(
                f"Skipping {idx} because it has no quantities, compute single point first."
            )


    print(len(mol_documents))
    print(len(mol_quantities))

    mol_docs = []

    for document, quantity in zip(mol_documents, mol_quantities):
        mol_docs.append([document, quantity])

    print(len(mol_docs))
    # print(mol_docs)

    print("CREATING MOL ENTRIES")
    mol_entries = list()
    for doc in mol_docs:
        try:
            task_id = doc[0]["task_id"]
            mol_entries.append(MoleculeEntry.from_molecule_document(doc[0]["output"], task_id))
        except MoleculeEntryError:
            print("skipping")
            continue
    print(len(mol_entries))
    print("FILTERING")
    mol_entries_filtered = remove_high_energy_mol_entries(mol_entries)
    good_task_ids = [entry.entry_id for entry in mol_entries_filtered]
    mol_docs = [doc for doc in mol_docs if doc[0]["task_id"] in good_task_ids]
    print(len(mol_entries_filtered))
    print(len(good_task_ids))
    print(len(mol_docs))

    print("BUILDING")
    operations = list()
    for mol_doc in mol_docs:
        entry = dict()
        # keys_list = list(mol_doc[0].keys())
        # print(keys_list)
        # keys_list2 = list(mol_doc[1].keys())
        # print(keys_list2)
        # print(mol_doc[])
        # print("\n")

        # # For now, don't allow any incomplete entry to be included
        # if ("bonds" not in mol_doc or "vibrational_frequencies" not in mol_doc) and len(mol_doc["molecule"]["sites"]) > 1:
        #     print("NO BONDS FOR MOLECULE WITH MORE THAN 1 ATOM: {}".format(mol_doc["task_id"]))
        #     continue

        # For now, don't allow any incomplete entry to be included
        if ("frequencies" not in mol_doc[0]["calcs_reversed"][0]) and len(mol_doc["molecule"]["sites"]) > 1:
            print("NO BONDS FOR MOLECULE WITH MORE THAN 1 ATOM: {}".format(mol_doc["task_id"]))
            continue

        # # For now, don't allow any incomplete entry to be included
        # if (len(mol_doc[0]["output"]["initial_molecule"]["sites"])) == 1:
        #     print("NO BONDS FOR MOLECULE WITH MORE THAN 1 ATOM: {}".format(mol_doc["task_id"]))
        #     continue

        # Molecule ID
        mol_id = "MgDME-" + str(mol_doc[0]["task_id"])
        entry["molecule_id"] = mol_id
        # print(entry)

        # Basic molecule information
        mol = Molecule.from_dict(mol_doc[0]["output"]["optimized_molecule"])
        # print(mol.keys())
        species = [str(e) for e in mol.species]
        coords = mol.cart_coords.tolist()
        entry["molecule"] = mol_doc[0]["output"]["optimized_molecule"]
        entry["charge"] = mol.charge
        entry["spin_multiplicity"] = mol.spin_multiplicity
        entry["species"] = species
        entry["xyz"] = coords
        # entry["bonds"] = mol_doc[1]["output"]["initial_molecule"].get("bonds", {"critic": list(), "nbo": list()})
        # entry["molecule_graph"] = MoleculeGraph.with_edges(
        #     mol,
        #     {tuple(i): {} for i in entry["bonds"]["critic"]}
        # ).as_dict()
        # print(entry["bonds"])

        molecule = mol_doc[0]["output"]["optimized_molecule"]
        molecule = Molecule.from_dict(molecule)
        molecule_graph = MoleculeGraph.with_local_env_strategy(molecule, OpenBabelNN())
        entry["molecule_graph"] = molecule_graph

        entry["number_atoms"] = len(mol_doc[0]["output"]["optimized_molecule"]["sites"])
        entry["number_elements"] = len(molecule_graph.molecule.composition.elements)
        entry["composition"] = molecule_graph.molecule.composition
        entry["formula_alphabetical"] = mol_doc[0]["formula_alphabetical"]
        entry["chemical_system"] = mol_doc[0]["chemsys"]

        elements_counts = re.findall(r'([A-Z][a-z]*)(\d*)', entry["formula_alphabetical"])
        composition = {}
        elements = []

        for element, count in elements_counts:
            if count:
                composition[element] = float(count)
            else:
                composition[element] = 1.0  # If no count is specified, assume 1
            elements.append(element)

        entry["composition"] = composition
        entry["elements"] = elements

        # Atomic partial charges
        entry["partial_charges"] = dict()
        entry["partial_charges"]["resp"] = list(mol_doc[0]["output"]["resp"])
        if mol.spin_multiplicity == 1:
            entry["partial_charges"]["mulliken"] = list(mol_doc[0]["output"]["mulliken"])
        else:
            entry["partial_charges"]["mulliken"] = [x[0] for x in mol_doc[0]["output"]["mulliken"]]
        if "critic" in mol_doc:
            entry["partial_charges"]["critic2"] = list(mol_doc["critic"]["processed"]["charges"])
        else:
            entry["partial_charges"]["critic2"] = None
        if "nbo" in mol_doc[1]["output"]:
            entry["partial_charges"]["nbo"] = [float(mol_doc[1]["output"]["nbo"]["natural_populations"][0]["Charge"][str(i)]) for i in range(len(Molecule.from_dict(mol_doc[1]["output"]["initial_molecule"])))]
        else:
            entry["partial_charges"]["nbo"] = None

        if mol.spin_multiplicity == 1:
            entry["partial_spins"] = {"mulliken": None, "nbo": None}
        else:
            entry["partial_spins"] = {"mulliken": [x[1] for x in mol_doc[1]["output"]["mulliken"]],
                                      "nbo": [float(mol_doc[1]["output"]["nbo"]["natural_populations"][0]["Density"][str(i)]) for i in range(len(Molecule.from_dict(mol_doc[1]["output"]["initial_molecule"])))]}

        nbo_warnings = mol_doc[0]["warnings"]

        # Vibrational information
        if len(mol) > 1:
            entry["vibration"] = dict()
            entry["vibration"]["frequencies"] = list(mol_doc[0]["calcs_reversed"][0]["frequencies"])
            entry["vibration"]["mode_vectors"] = list(mol_doc[0]["calcs_reversed"][0]["frequency_mode_vectors"])
            entry["vibration"]["IR_intensities"] = list(mol_doc[0]["calcs_reversed"][0]["IR_intens"])
        else:
            # Doesn't make sense for single atoms
            entry["vibration"] = None

        # Molecular thermodynamics
        entry["thermo"] = {"raw": dict(),
                           "eV": dict(),
                           "quasi_rrho_eV": dict(),
                           "shifted_rrho_eV": dict()}

        entry["thermo"]["raw"]["electronic_energy_Ha"] = mol_doc[0]["output"]["final_energy"]
        entry["thermo"]["raw"]["total_enthalpy_kcal/mol"] = mol_doc[0]["calcs_reversed"][0]["total_enthalpy"]
        entry["thermo"]["raw"]["total_entropy_cal/molK"] = mol_doc[0]["calcs_reversed"][0]["total_entropy"]
        if len(mol) > 1:
            entry["thermo"]["raw"]["translational_enthalpy_kcal/mol"] = mol_doc[0]["calcs_reversed"][0]["trans_enthalpy"]
            entry["thermo"]["raw"]["rotational_enthalpy_kcal/mol"] = mol_doc[0]["calcs_reversed"][0]["rot_enthalpy"]
            entry["thermo"]["raw"]["vibrational_enthalpy_kcal/mol"] = mol_doc[0]["calcs_reversed"][0]["vib_enthalpy"]
            entry["thermo"]["raw"]["translational_entropy_cal/molK"] = mol_doc[0]["calcs_reversed"][0]["trans_entropy"]
            entry["thermo"]["raw"]["rotational_entropy_cal/molK"] = mol_doc[0]["calcs_reversed"][0]["rot_entropy"]
            entry["thermo"]["raw"]["vibrational_entropy_cal/molK"] = mol_doc[0]["calcs_reversed"][0]["vib_entropy"]
        else:
            entry["thermo"]["raw"]["translational_enthalpy_kcal/mol"] = mol_doc[0]["calcs_reversed"][0]["total_enthalpy"]
            entry["thermo"]["raw"]["rotational_enthalpy_kcal/mol"] = None
            entry["thermo"]["raw"]["vibrational_enthalpy_kcal/mol"] = None
            entry["thermo"]["raw"]["translational_entropy_cal/molK"] = mol_doc[0]["calcs_reversed"][0]["total_entropy"]
            entry["thermo"]["raw"]["rotational_entropy_cal/molK"] = None
            entry["thermo"]["raw"]["vibrational_entropy_cal/molK"] = None

        entry["thermo"]["eV"]["electronic_energy"] = entry["thermo"]["raw"]["electronic_energy_Ha"] * ase_units.Hartree
        entry["thermo"]["eV"]["total_enthalpy"] = entry["thermo"]["raw"]["total_enthalpy_kcal/mol"] * kcal_per_mol_to_eV
        entry["thermo"]["eV"]["total_entropy"] = entry["thermo"]["raw"]["total_entropy_cal/molK"] * kcal_per_mol_to_eV / 1000
        entry["thermo"]["eV"]["free_energy"] = entry["thermo"]["eV"]["electronic_energy"] + entry["thermo"]["eV"]["total_enthalpy"] - 298.15 * entry["thermo"]["eV"]["total_entropy"]

        pga = PointGroupAnalyzer(mol)
        entry["point_group"] = pga.sch_symbol

        rotational_symmetry_numbers = {1: ["C1", "Cs", "Ci", "C*v", "S2"],
                                       2: ["C2", "C2h", "C2v", "S4", "D*h"],
                                       3: ["C3", "C3h", "C3v", "S6"],
                                       4: ["C4v", "D4h", "D4d", "D2", "D2h", "D2d"],
                                       5: ["C5v", "Ih"],
                                       6: ["D3", "D3h", "D3d"],
                                       10: ["D5h", "D5d"],
                                       12: ["T", "Td", "Th", "D6h"],
                                       14: ["D7h"],
                                       16: ["D8h"],
                                       24: ["Oh"],
                                       np.inf: ["Kh"]}

        # Skip for single atoms
        if len(mol) > 1:
            r = 1
            for rot_num, point_groups in rotational_symmetry_numbers.items():
                if entry["point_group"] in point_groups:
                    r = rot_num
                    break
            if entry["point_group"] in ["C*v", "D*h"]:
                linear = True
            else:
                linear = False

            rrho_input = {"mol": mol,
                          "mult": mol.spin_multiplicity,
                          "frequencies": entry["vibration"]["frequencies"],
                          "elec_energy": entry["thermo"]["raw"]["electronic_energy_Ha"]}

            try:
                qrrho = QuasiRRHO(rrho_input, sigma_r=r, linear=linear)
                entry["thermo"]["quasi_rrho_eV"]["electronic_energy"] = entry["thermo"]["eV"]["electronic_energy"]
                entry["thermo"]["quasi_rrho_eV"]["total_enthalpy"] = qrrho.enthalpy_quasiRRHO * 27.2114
                entry["thermo"]["quasi_rrho_eV"]["total_entropy"] = qrrho.entropy_quasiRRHO * 27.2114
                entry["thermo"]["quasi_rrho_eV"]["free_energy"] = qrrho.free_energy_quasiRRHO * 27.2114
            except ZeroDivisionError:
                entry["thermo"]["quasi_rrho_eV"] = None

            try:
                rrho_shifted = ShiftedRRHO(rrho_input, sigma_r=r, linear=linear)
                entry["thermo"]["shifted_rrho_eV"]["electronic_energy"] = entry["thermo"]["eV"]["electronic_energy"]
                entry["thermo"]["shifted_rrho_eV"]["total_enthalpy"] = rrho_shifted.enthalpy_RRHO * 27.2114
                entry["thermo"]["shifted_rrho_eV"]["total_entropy"] = rrho_shifted.entropy_RRHO * 27.2114
                entry["thermo"]["shifted_rrho_eV"]["free_energy"] = rrho_shifted.free_energy_RRHO * 27.2114
            except ZeroDivisionError:
                entry["thermo"]["shifted_rrho_eV"] = None
        else:
            entry["thermo"]["quasi_rrho_eV"] = None
            entry["thermo"]["shifted_rrho_eV"] = None

        entry["redox"] = {"electron_affinity_eV": None,
                          "ionization_energy_eV": None}
        # if mol_doc[0]["vertical_ea"] is not None:
        #     entry["redox"]["electron_affinity_eV"] = mol_doc["vertical_ea"] * 27.2114
        # if mol_doc[0]["vertical_ie"] is not None:
        #     entry["redox"]["ionization_energy_eV"] = mol_doc["vertical_ie"] * 27.2114

        entries.append(entry)

    dumpfn(entries, "Mg_DME_mol_entries_3_27.json")
    print("entries length")
    print(len(entries))

    cleaned_entries = loads(open("Mg_DME_mol_entries_3_27.json").read())

    for entry in cleaned_entries:
        operations.append(
            UpdateOne(
                {"molecule_id": entry["molecule_id"]},
                {"$set": entry},
                upsert=True
            )
        )

    print("WRITING {} ENTRIES".format(len(operations)))
    # db.db["tasks"].bulk_write(operations)
    print("COMPLETED")

