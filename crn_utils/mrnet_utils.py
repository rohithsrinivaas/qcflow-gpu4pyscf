from mrnet.core.mol_entry import MoleculeEntry, MoleculeEntryError
from mrnet.core.reactions import bucket_mol_entries
from typing import List


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

