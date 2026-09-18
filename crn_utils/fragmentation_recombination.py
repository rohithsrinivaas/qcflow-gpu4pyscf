"""Create fragments for selected molecules."""
# Originally authored by: Peichen Zhong
# Additional modifications: Nagappan Nataraja Chockalingam, Yinhe Wang

import os
import logging
import json
import itertools
import numpy as np
import copy
from collections import defaultdict
from typing import List, Any

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.fragmenter import Fragmenter
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import OpenBabelNN

from monty.serialization import loadfn

from ase import io as ase_io
from ase import data as ase_data
from copy import deepcopy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mkdir(path: str):
    """Make directory.

    Args:
        path (str): directory name

    Returns:
        path
    """
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print("Folder exists")
    return path


class FragmentReconnect:

    MAX_ELECTRONS = 180

    BOND_MAX = {
        "C": 4,
        "P": 5,
        "S": 6,
        "O": 2,
        "N": 3,
        "B": 5,
        "Cl": 1,
        "F": 1,
    }
    AXIS_OF_ROTATION = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]

    delta_distance = 0.05

    groupname = "initial_structures_fragmentation_recombination"

    def __init__(
        self,
        initial_graphs_collection,
        molecule_list: List[Molecule],
        depth: int = 1,
        bonding_factor_max: float = 1.5,
        bonding_factor_min: float = 0.5,
        bonding_factor_number: int = 10,
        number_of_angles: int = 100,
        **kwargs: Any,
    ):
        """Create fragments and recominations for a list of initial molecule graphs.

        Args:
            initial_graphs_collection: MongoDB collection where the results will be stored
            molecule_list: List of initial molecule graphs
            depth: Depth of the fragmentation
            bonding_factor_max: Maximum bonding factor (default: 1.5)
            bonding_factor_min: Minimum bonding factor (default: 0.5)
            bonding_factor_number: Number of bonding factors (default: 10)
            number_of_angles: Number of angles to rotate the fragments (default: 10)
        """
        self.call_counter = 0
        self.molecule_list = molecule_list
        self.depth = depth
        self.bonding_factor_max = bonding_factor_max
        self.bonding_factor_min = bonding_factor_min
        self.bonding_factor_number = bonding_factor_number
        self.number_of_angles = number_of_angles
        self.initial_graphs_collection = initial_graphs_collection

        self.DEBUG = kwargs.get("debug", False)

        self.output_folder = kwargs.get(
            "output_folder", os.path.join("outputs", "initial_structures")
        )
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def __call__(self, *args: Any, **kwgs: Any) -> Any:
        self._get_graphs_already_in_database()
        self.get_fragments()
        self.get_recombinants()

    def run(self, if_add_monoatomic = False, min_idx=0, shard=None):
        self._get_graphs_already_in_database()
        self.get_fragments(if_add_monoatomic)
        self.get_recombinants(min_idx=min_idx, shard=shard)

    def debug(self, *args: Any, **kwgs: Any) -> Any:
        # self._get_graphs_already_in_database()
        self.get_fragments()
        # self.get_recombinants()

    @staticmethod
    def _dedup_key(molecule_graph):
        """Cheap invariant that isomorphic graphs must share, used to bucket
        the dedup database so a duplicate check only compares against
        same-formula/charge/spin candidates instead of every stored graph.
        """
        mol = molecule_graph.molecule
        return (mol.composition.formula, mol.charge, mol.spin_multiplicity)

    def _get_graphs_already_in_database(self):
        """Get the graphs already in the database, bucketed by _dedup_key."""
        self.graphs_already_in_database = defaultdict(list)
        n = 0
        for graph in self.initial_graphs_collection.find(
            {"tags.group": self.groupname}
        ):
            graph.pop("tags")
            mol_graph = MoleculeGraph.from_dict(graph)
            self.graphs_already_in_database[self._dedup_key(mol_graph)].append(mol_graph)
            n += 1
        logger.info(f"Number of graphs already in database: {n}")

    def _check_already_in_database(self, molecule_graph):
        """Check if the molecule graph is already in the database.

        Only scans the bucket matching this graph's (formula, charge, spin) --
        with tens of thousands of stored graphs, a full linear isomorphism
        scan against every one of them (the previous behavior) dominates
        runtime; isomorphic graphs are guaranteed to share this key, so
        bucketing is exact, not an approximation.
        """
        bucket = self.graphs_already_in_database.get(self._dedup_key(molecule_graph), [])
        for graph in bucket:
            if molecule_graph.isomorphic_to(graph):
                return True
        return False


    def store_xyz_fragments(self, label: str, molecule_graphs: List[Molecule]):
        """Store fragments as xyz files."""

        all_atoms = []
        for idx, molecule_graph in enumerate(molecule_graphs):
            atoms = AseAtomsAdaptor.get_atoms(molecule_graph.molecule)
            all_atoms.append(atoms)
        ase_io.write(os.path.join(self.output_folder, f"{label}.xyz"), all_atoms)

    def get_fragments_old(self):

        """
        TODO: change the name into varaibles
        
        Generate fragments for all molecules in the list."""

        # Create a list of fragmented molecule graphs
        self.frag_molecule_graphs = []
        # Store the connecting atoms
        self.idx_connecting_molecules = []

        for idx, molecule in enumerate(self.molecule_list):
            logger.info("Generating fragments for molecule %s", idx)
            logger.info(f"Fragmenting molecule up to depth {self.depth}")
            fragmenter = Fragmenter(molecule, depth=self.depth, open_rings=True)
            logger.info(f"Number of fragments: {fragmenter.total_unique_fragments}")

            for key, molecule_graphs in fragmenter.unique_frag_dict.items():
                label = "fragments_" + "_".join(key.split())
                self.store_xyz_fragments(label, molecule_graphs)
                self.frag_molecule_graphs.extend(molecule_graphs)

                # Determine the connecting atoms in the molecule
                for idx_mol, molecule_graph in enumerate(molecule_graphs):
                    _idx_connecting_atoms = self._generate_connecting_atoms(
                        molecule_graph
                    )
                    tags = {
                        "group": self.groupname,
                        "type_structure": "fragment",
                        "parent_molecule": molecule_graph.molecule.as_dict(),
                        "idx_connecting_atoms": _idx_connecting_atoms,
                        "idx_mol": idx_mol,
                    }
                    self._put_graph_in_database(molecule_graph, tags)

        # # add the monoatomic species into the fragmented molecule list
        # mol_graph = get_molecule_graph_from_json("mvbe-980419", "mg_initial_structures_12_06_2023.json")
        # self.frag_molecule_graphs.extend(mol_graph)
        # tags = {
        #     "group": self.groupname,
        #     "type_structure": "fragment_monoatomic",
        #  }
        # #self._put_monoatomic_graph_in_database(mol_graph, tags)
        # # _idx_connecting_atoms = self._generate_connecting_atoms(mol_graph)
        # # self.idx_connecting_molecules.append(_idx_connecting_atoms)
        # self.idx_connecting_molecules.append([0])
        # mol_graph = get_molecule_graph_from_json("mvbe-855003", "mg_initial_structures_12_06_2023.json")
        # self.frag_molecule_graphs.extend(mol_graph)
        # # _idx_connecting_atoms = self._generate_connecting_atoms(mol_graph)
        # # self.idx_connecting_molecules.append(_idx_connecting_atoms)
        # self.idx_connecting_molecules.append([0])

        # mol_graph = get_molecule_graph_from_json("mvbe-320736", "mg_initial_structures_12_06_2023.json")
        # self.frag_molecule_graphs.extend(mol_graph)
        # # _idx_connecting_atoms = self._generate_connecting_atoms(mol_graph)
        # # self.idx_connecting_molecules.append(_idx_connecting_atoms)
        # self.idx_connecting_molecules.append([0])

        # mol_graph = get_molecule_graph_from_json("mvbe-877247", "mg_initial_structures_12_06_2023.json")
        # self.frag_molecule_graphs.extend(mol_graph)
        # # _idx_connecting_atoms = self._generate_connecting_atoms(mol_graph)
        # # self.idx_connecting_molecules.append(_idx_connecting_atoms)
        # self.idx_connecting_molecules.append([0])

        # mol_graph = get_molecule_graph_from_json("mvbe-487010", "mg_initial_structures_12_06_2023.json")
        # self.frag_molecule_graphs.extend(mol_graph)
        # # _idx_connecting_atoms = self._generate_connecting_atoms(mol_graph)
        # # self.idx_connecting_molecules.append(_idx_connecting_atoms)
        # self.idx_connecting_molecules.append([0])

        print(len(self.frag_molecule_graphs))
        print(len(self.idx_connecting_molecules))
        print(self.idx_connecting_molecules)

    def is_duplicate_in_fragments(self, mol_graph):
        for existing_graph in self.frag_molecule_graphs:
            if mol_graph.isomorphic_to(existing_graph):
                if mol_graph.molecule.charge == existing_graph.molecule.charge:
                    if mol_graph.molecule.spin_multiplicity == existing_graph.molecule.spin_multiplicity:
                        return True
        return False
    
    def get_fragments(self, if_add_monoatomic = False):

        """        
        Generate fragments for all molecules in the list.
        """

        # Create a list of fragmented molecule graphs
        self.frag_molecule_graphs = []
        # Store the connecting atoms
        self.idx_connecting_molecules = []

        for idx, molecule in enumerate(self.molecule_list):

            if (len(molecule) == 1) and (if_add_monoatomic):
                logger.info(f"Adding molecule {idx} with only one atom")
                molecule_graph = MoleculeGraph.from_local_env_strategy(molecule, strategy= OpenBabelNN())
                tags = {
                        "group": self.groupname,
                        "type_structure": "fragment_monoatomic",
                        "parent_molecule": molecule_graph.molecule.as_dict(),
                        "idx_connecting_atoms": 0,
                        "idx_mol": 0,
                    }
                label = "fragments_" + "_".join(molecule.formula.split())

                self.frag_molecule_graphs.extend([molecule_graph])
                self.idx_connecting_molecules.append([0])

                #self._put_graph_in_database(molecule_graph, tags)

                for charge in [-2, -1, 0, 1, 2]:
                    mol_graph_variantf = deepcopy(molecule_graph)
                    mol_graph_variantf.molecule.set_charge_and_spin(charge)
                    tags = {
                            "group": self.groupname,
                            "type_structure": "fragment",
                            "parent_molecule": molecule_graph.molecule.as_dict(),
                            "idx_connecting_atoms": _idx_connecting_atoms,
                            "idx_mol": idx_mol,
                            "charge_variant":charge
                                }
                    self._put_graph_in_database(mol_graph_variantf, tags)

            else:
                logger.info("Generating fragments for molecule %s", idx)
                logger.info(f"Fragmenting molecule up to depth {self.depth}")
                fragmenter = Fragmenter(molecule, depth=self.depth, open_rings=True,use_metal_edge_extender=True)
                logger.info(f"Number of fragments: {fragmenter.total_unique_fragments}")

                # === ADD ORIGINAL MOLECULE AS FRAGMENT ===
                original_mol_graph = MoleculeGraph.from_local_env_strategy(molecule, strategy=OpenBabelNN())
                idx_connecting_atomso = self._generate_connecting_atoms(original_mol_graph)
                self.frag_molecule_graphs.append(original_mol_graph)
                #print("Connecting atoms for original: ",idx_connecting_atomso)
                # Tag it appropriately
                tags = {
                    "group": self.groupname,
                    "type_structure": "fragment_original",
                    "parent_molecule": original_mol_graph.molecule.as_dict(),
                    "idx_connecting_atoms": idx_connecting_atomso,
                    "idx_mol": 0,  # or any ID you want to assign
                    }
                # Store and track
                #self._put_graph_in_database(original_mol_graph, tags)

                for charge in [-2, -1, 0, 1, 2]:
                    mol_graph_variantf = deepcopy(original_mol_graph)
                    mol_graph_variantf.molecule.set_charge_and_spin(charge)
                    tags["charge_variant"] = charge
                    self._put_graph_in_database(mol_graph_variantf, tags)
                
                #fragments generated in this step
                for key, molecule_graphs in fragmenter.unique_frag_dict.items():

                    # Keep only the graphs in this group that aren't already
                    # tracked, then process each one exactly once (previously
                    # this extended/reprocessed the *entire* molecule_graphs
                    # list once per non-duplicate member, multiplying the
                    # fragment count and stored charge variants by up to
                    # len(molecule_graphs)).
                    unique_molecule_graphs = [
                        mol_graph for mol_graph in molecule_graphs
                        if not self.is_duplicate_in_fragments(mol_graph)
                    ]
                    if not unique_molecule_graphs:
                        print("$$$$$$$$$ Fragment already present $$$$$$$$$$$")
                        continue

                    self.frag_molecule_graphs.extend(unique_molecule_graphs)

                    # Determine the connecting atoms in the molecule
                    for idx_mol, molecule_graph in enumerate(unique_molecule_graphs):
                        _idx_connecting_atoms = self._generate_connecting_atoms(
                            molecule_graph
                        )

                        for charge in [-2, -1, 0, 1, 2]:
                            label = "fragments_"+"_".join(key.split())+f"charge_{charge}"
                            mol_graph_variantf = deepcopy(molecule_graph)
                            mol_graph_variantf.molecule.set_charge_and_spin(charge)
                            self.store_xyz_fragments(label, [mol_graph_variantf])
                            tags = {
                            "group": self.groupname,
                            "type_structure": "fragment",
                            "parent_molecule": molecule_graph.molecule.as_dict(),
                            "idx_connecting_atoms": _idx_connecting_atoms,
                            "idx_mol": idx_mol,
                            "charge_variant":charge
                                }

                            self._put_graph_in_database(mol_graph_variantf, tags)

        print(len(self.frag_molecule_graphs))
        print(len(self.idx_connecting_molecules))
        print(self.idx_connecting_molecules)
        print(sum(len(v) for v in self.graphs_already_in_database.values()))


    def _generate_connecting_atoms(self, frag_mol_graph):
        """For all fragments, generate a list of connecting atoms."""
        molecule = frag_mol_graph.molecule
        natoms_molecule = len(molecule)
        _idx_connecting_atoms = []
        for idx in range(natoms_molecule):
            if not self._species_test(molecule, idx):
                continue
            if not self._bond_test(
                frag_mol_graph.get_connected_sites(idx), str(molecule[idx].specie)
            ):
                continue
            _idx_connecting_atoms.append(idx)
        self.idx_connecting_molecules.append(_idx_connecting_atoms)
        return _idx_connecting_atoms

    def _put_graph_in_database(self, molecule_graph, tags: dict):
        """Put the structure in the initial structure database."""

        if self._check_already_in_database(molecule_graph):
            logging.warning("Structure already present in database")
            print("Structure already present in database")
            return

        if self.DEBUG:
            logger.warning("Debug calculation, not storing molecule in database")
            return

        molecule_graph_dict = molecule_graph.as_dict()
        serialize_molecule_graph(molecule_graph_dict)
        molecule_graph_dict["tags"] = tags
        self.initial_graphs_collection.insert_one(molecule_graph_dict)
        self.graphs_already_in_database[self._dedup_key(molecule_graph)].append(molecule_graph)

    def _put_monoatomic_graph_in_database(self, molecule_graph, tags: dict):
        molecule_graph_dict = molecule_graph.as_dict()
        serialize_molecule_graph(molecule_graph_dict)
        molecule_graph_dict["tags"] = tags
        self.initial_graphs_collection.insert_one(molecule_graph_dict)
        self.graphs_already_in_database[self._dedup_key(molecule_graph)].append(molecule_graph)

    def check_charge(self, frag1, frag2) -> bool:
        """Check if the charge passes the criteria."""
        tot_charge = frag1.molecule.charge + frag2.molecule.charge
        if tot_charge not in [-2, -1, 0, 1, 2]:
            logging.debug(
                f"Charge {tot_charge}e with {frag1.molecule.charge} and {frag2.molecule.charge} not in [-2, -1, 0, 1, 2]"
            )
            return False
        else:
            logging.debug(
                f"Charge: {tot_charge}e with {frag1.molecule.charge} and {frag2.molecule.charge}  is accepted"
            )
            return True

    def check_number_of_electrons_max(self, frag1, frag2) -> bool:
        """Check if the number of electrons exceeds our computational capacity."""
        tot_electrons = frag1.molecule.nelectrons + frag2.molecule.nelectrons
        if tot_electrons < self.MAX_ELECTRONS:
            logging.debug(f"Electrons {tot_electrons} is accepted")
            return True
        else:
            return False

    def _species_test(self, molecule, idx):
        """Check if the species is suitable to be added"""
        if str(molecule[idx].specie) in self.BOND_MAX:
            return True
        elif str(molecule[idx].specie) in ["H", "Li", "Mg", "Ca"]:
            # Also pass isolated instances of H, Li, Mg, Ca
            if len(molecule) == 1:
                return True
        else:
            return False

    def _bond_test(self, connected_sites, element):
        """Check if the atom has enough free bonds."""

        if element in ["H", "Li", "Mg", "Ca"]:
            return True

        # Check if the atom has enough free bonds
        tot_connected_sites = len(connected_sites)

        # Remove metal sites from the tot_connected_sites
        for k, site in enumerate(connected_sites):
            if str(site.site.specie) in ["Li", "Mg", "Ca"]:
                tot_connected_sites -= 1

        if tot_connected_sites < self.BOND_MAX[element]:
            return True
        else:
            return False
        

    def get_recombinants_multiple(self, max_fragments = 3):
        num_fragments = len(self.frag_molecule_graphs)
        idx_fragments = list(range(num_fragments))

        for num_fragments in range(2, max_fragments + 1):
            combinations = itertools.combinations(idx_fragments, num_fragments)
            for combination in combinations:
                fragments = [self.frag_molecule_graphs[i] for i in combination]
                
                if not self.check_charge(fragments):
                    continue
                if not self.check_number_of_electrons_max(fragments):
                    continue

                connecting_atoms = [self.idx_connecting_molecules[i] for i in combination]
                combined_mol_graph = self.combine_multiple_mol_graphs(fragments, connecting_atoms)
                
                if combined_mol_graph is not None:
                    # Process the combined molecule graph
                    # (Store it, add tags, etc.)
                    raise NotImplementedError("Processing of combined molecule graph not implemented yet")

        # Rest of the method...

    
    def get_recombinants(self, min_idx=0, shard=None):
        """
        Create recombinations of the molecule.

        min_idx: skip pair-indices (the flat enumeration over
        combinations_with_replacement) below this -- used to resume a run
        from a known stopping point without redoing completed work.
        shard: optional (worker_id, num_workers) -- when given, this worker
        only processes pair-indices where idx % num_workers == worker_id.
        Round-robin rather than contiguous ranges, because per-pair cost is
        highly variable (some pairs reject instantly on the charge/electron
        filter, others require an expensive docking search), so a contiguous
        chunk risks clustering the expensive ones onto one worker.
        """
        num_fragments = len(self.frag_molecule_graphs)
        idx_fragments = list(range(num_fragments))
        # Generate the combinations of fragments
        combinations = itertools.combinations_with_replacement(idx_fragments, 2)

        #Iterate over the combinations
        tot_number_of_recombinations = 0
        accepted_number_of_recombinations = 0

        for idx, combination in enumerate(combinations):
            #print(combination)

            if idx < min_idx:
                continue
            if shard is not None:
                worker_id, num_workers = shard
                if idx % num_workers != worker_id:
                    continue

            # Get the two fragments
            frag1 = self.frag_molecule_graphs[combination[0]]
            frag2 = self.frag_molecule_graphs[combination[1]]

            # Check charge
            if not self.check_charge(frag1, frag2):
                continue

            if not self.check_number_of_electrons_max(frag1, frag2):
                continue
            
            # Iterate over the connecting atoms and connect the two graphs
            # at the appropriate connecting atom
            heavy_atoms_frag1 = self.idx_connecting_molecules[combination[0]]
            heavy_atoms_frag2 = self.idx_connecting_molecules[combination[1]]
            
            # Iterate over the potential connecting atoms
            for idx1 in heavy_atoms_frag1:
                for idx2 in heavy_atoms_frag2:
                    # Add to the total number of recombinations
                    tot_number_of_recombinations += 1
                    
                    # Create a new graph with both fragments
                    combined_mol_graph = self.combine_mol_graph(
                        frag1, frag2, idx, idx1, idx2
                    )
                    
                    if combined_mol_graph is not None:
                        idx1_mol = idx1
                        idx2_mol = idx2 + len(frag1.molecule)
                        
                        # Generate the tags to store in mongodb
                        frag1_dict = frag1.as_dict()
                        frag2_dict = frag2.as_dict()
                        serialize_molecule_graph(frag1_dict)
                        serialize_molecule_graph(frag2_dict)
                        
                        combined_molecule = combined_mol_graph.molecule
                        
                        for charge in [-2, -1, 0, 1, 2]:
                            label = f"combination_{idx}_link_{idx1_mol}_link_{idx2_mol}_charge_{charge}"
                            accepted_number_of_recombinations = accepted_number_of_recombinations + 1
                            
                            mol_graph_variant = deepcopy(combined_mol_graph)
                            mol_graph_variant.molecule.set_charge_and_spin(charge)
                            
                            self.store_xyz_fragments(label, [mol_graph_variant])
                            
                            tags = {
                                "group": self.groupname,
                                "type_structure": "recombination",
                                "frag1_graphs": frag1_dict,
                                "connecting_atoms_frag1": heavy_atoms_frag1,
                                "idx1_connecting_atoms": idx1,
                                "frag2_graphs": frag2_dict,
                                "connecting_atoms_frag2": heavy_atoms_frag2,
                                "idx2_connecting_atoms": idx2,
                                "charge_variant": charge
                            }
                            self._put_graph_in_database(mol_graph_variant, tags)
                            
        # (Indentation Fix: Logging should happen *after* all loops are finished)
        logger.info(f"Total number of recombinations: {tot_number_of_recombinations}")
        logger.info(
            f"Accepted number of recombinations: {accepted_number_of_recombinations}"
        )

    def vdw_radii(self, symbol):
        """Repurpose ASEs vdw radii."""
        atomic_number = ase_data.atomic_numbers[symbol]
        return ase_data.vdw_radii[atomic_number]

    def covalent_radii(self, symbol):
        """Repurpose ASEs covalent radii."""
        atomic_number = ase_data.atomic_numbers[symbol]
        return ase_data.covalent_radii[atomic_number]

    def _check_overlap(
        self, mol_1, mol_2, bonding_factor, check_min=False, between_idx=None
    ):
        """Check if the two molecules overlap."""
        # Get the positions of the atoms
        positions1 = mol_1.cart_coords
        positions2 = mol_2.cart_coords

        species1 = mol_1.species
        species2 = mol_2.species
        species1 = [a.symbol for a in species1]
        species2 = [a.symbol for a in species2]

        # Generate the sum over vdw radii matrix
        dist_matrix = np.zeros((positions1.shape[0], positions2.shape[0]))
        radii_matrix = np.zeros((positions1.shape[0], positions2.shape[0]))

        # Iterate over the atoms to populate the matrix
        for i, specie1 in enumerate(species1):
            for j, specie2 in enumerate(species2):
                dist_matrix[i, j] = np.linalg.norm(positions1[i] - positions2[j])
                if i != between_idx[0] and j != between_idx[1]:
                    # If it is not between the two atoms, use the covalent
                    # radii to check for overlap
                    radii_matrix[i, j] = self.covalent_radii(
                        specie1
                    ) + self.covalent_radii(specie2)
                else:
                    # This is the bond between the two atoms, use the covlent
                    # radii / 2
                    radii_matrix[i, j] = (
                        self.covalent_radii(specie1) + self.covalent_radii(specie2)
                    ) / bonding_factor

        # If the distance is smaller than the sum of the vdw radii, there is an overlap
        if np.any(dist_matrix < radii_matrix):
            return True

        # Check that the maximum distance between the indices in between_idx is
        # largest of all the distances between the two atoms
        if check_min:
            expect_min_distance = dist_matrix[between_idx[0], between_idx[1]]
            # Sort the distance matrix from lowest to highest
            dist_matrix_sorted = np.sort(dist_matrix.flatten(), axis=None)
            # Get up to the 10 lowest distances
            current_min_distance = dist_matrix_sorted[
                0 : min(10, len(dist_matrix.flatten()))
            ]
            if np.any(expect_min_distance <= current_min_distance):
                return False
            else:
                # There is an atom closer that the indexed atoms.
                logging.debug(
                    f"Current min distance: {current_min_distance} is lower than expected min distance: {expect_min_distance}"
                )
                return True
            # else:
            #     return False
        else:
            return False

    def rotate_molecules(
        self,
        copy_1: MoleculeGraph,
        copy_2: MoleculeGraph,
        axis: List[float],
        angle: float,
        bonding_factor: float,
        intermediate_molecules: List[Molecule],
        idx1: int,
        idx2: int,
        rotate_first: bool = True,
    ):

        copy_1_copy = copy.deepcopy(copy_1)
        copy_2_copy = copy.deepcopy(copy_2)

        # Rotate the first molecule first
        if rotate_first:
            rotation_indices = list(range(len(copy_1_copy.molecule)))
            copy_1_copy.molecule.rotate_sites(
                indices=rotation_indices,
                theta=angle,
                axis=axis,
                anchor=[0, 0, 0],
            )
        else:
            rotation_indices = list(range(len(copy_2_copy.molecule)))
            copy_2_copy.molecule.rotate_sites(
                indices=rotation_indices,
                theta=angle,
                axis=axis,
                anchor=[0, 0, 0],
            )

        # Store the generated molecule based on copy_2
        inter_copy_1 = copy.deepcopy(copy_1_copy)
        for site in copy_2_copy.molecule:
            inter_copy_1.molecule.append(site.specie, site.coords)
        intermediate_molecules.append(inter_copy_1)

        if not self._check_overlap(
            copy_1_copy.molecule,
            copy_2_copy.molecule,
            bonding_factor,
            check_min=True,
            between_idx=[idx1, idx2],
        ):
            return True, copy_1_copy, copy_2_copy
        else:
            return False, None, None

    def combine_mol_graph(self, molgraph_1, molgraph_2, idx, idx1, idx2):
        """Combine molecular graphs and generate the starting structures for a DFT calculation."""

        # Commonly used information in this method
        mol_1 = molgraph_1.molecule
        mol_2 = molgraph_2.molecule
        cspec_1 = str(mol_1[idx1].specie)
        cspec_2 = str(mol_2[idx2].specie)
        sum_radii = self.covalent_radii(cspec_1) + self.covalent_radii(cspec_2)

        # Start by moving the first molecule to the origin based on the coordinates
        # of the connecting atom in the molecule.
        copy_1 = copy.deepcopy(molgraph_1)
        copy_1.molecule.translate_sites(
            list(range(len(molgraph_1.molecule))),
            vector=-molgraph_1.molecule[idx1].coords,
        )

        # Get the index of the connecting atoms
        idx1_mol = idx1
        idx2_mol = idx2 + len(copy_1.molecule)
        label = f"intermediate_combination_{idx}_link_{idx1_mol}_link_{idx2_mol}"

        # Store the intermediate molecules during the rotation
        intermediate_molecules = []
        found_structure = False

        # Iterate over the placement of the atoms and the angle of rotation
        # to find feasible structures.
        range_bonding_pos = np.linspace(
            self.bonding_factor_max, self.bonding_factor_min, self.bonding_factor_number
        )
        #logging.debug(f"Range bonding: {range_bonding}")
        range_bonding_neg = np.linspace(
            -self.bonding_factor_max, -self.bonding_factor_min, self.bonding_factor_number
        )
        range_bonding = np.concatenate((range_bonding_pos, range_bonding_neg)).tolist() 
        logging.debug(f"Range bonding: {range_bonding}")

        for bonding_factor in range_bonding:

            # Generate the second molecule at the correct distance and angle
            translated_dist = (
                -np.array(molgraph_2.molecule[idx2].coords)
                + sum_radii / bonding_factor
                + self.delta_distance
            )

            copy_2 = copy.deepcopy(molgraph_2)
            copy_2.molecule.translate_sites(
                list(range(len(copy_2.molecule))),
                vector=translated_dist,
            )

            # Now iterate over the angles of rotation to find the correct angle
            # which does not overlap with the first molecule
            for axis in self.AXIS_OF_ROTATION:

                logging.debug(f"Rotate along axis: {axis}")
                # Iterate over angles.
                for angle in np.linspace(0, 2 * np.pi, self.number_of_angles):

                    angle_in_degrees = np.degrees(angle)
                    logging.debug(f"Rotating molecule to {angle_in_degrees} degrees")

                    # Rotate the first molecule first
                    found_structure, copy_1_new, copy_2_new = self.rotate_molecules(
                        copy_1=copy_1,
                        copy_2=copy_2,
                        axis=axis,
                        angle=angle,
                        bonding_factor=bonding_factor,
                        intermediate_molecules=intermediate_molecules,
                        idx1=idx1,
                        idx2=idx2,
                        rotate_first=True,
                    )

                    if found_structure:
                        copy_1 = copy_1_new
                        copy_2 = copy_2_new
                        break
                    else:
                        logging.debug(
                            f"No rotation found for first molecule rotation that does not overlap for axis {axis}"
                        )
                    found_structure, copy_1_new, copy_2_new = self.rotate_molecules(
                        copy_1=copy_1,
                        copy_2=copy_2,
                        axis=axis,
                        angle=angle,
                        bonding_factor=bonding_factor,
                        intermediate_molecules=intermediate_molecules,
                        idx1=idx1,
                        idx2=idx2,
                        rotate_first=False,
                    )

                    if found_structure:
                        copy_1 = copy_1_new
                        copy_2 = copy_2_new
                        break
                    else:
                        logging.debug(
                            f"No rotation found for second molecule rotation that does not overlap for axis {axis}"
                        )

                if found_structure:
                    break

            if found_structure:
                break
            else:
                logging.debug(
                    f"No displacement by {bonding_factor} leads to no overlap."
                )
        else:
            # No operation worked, write out the intermediate molecules
            label = "failed_" + label
            self.store_xyz_fragments(label, intermediate_molecules)
            logger.warning(
                f"Could not find a rotation angle that minimizes overlap for {label}"
            )
            return

        if len(intermediate_molecules) > 1:
            self.store_xyz_fragments(label, intermediate_molecules)

        # Finally, combine the two molecules
        for site in copy_2.molecule:
            copy_1.insert_node(len(copy_1.molecule), site.specie, site.coords)
        for edge in copy_2.graph.edges():
            side_1 = edge[0] + len(molgraph_1.molecule)
            side_2 = edge[1] + len(molgraph_1.molecule)
            copy_1.add_edge(side_1, side_2,weight=copy_2.graph.get_edge_data(edge[0],edge[1])[0]['weight'])
        copy_1.molecule.set_charge_and_spin(
            molgraph_1.molecule.charge + molgraph_2.molecule.charge
        )
        copy_1.add_edge(idx1, idx2 + len(molgraph_1.molecule), weight=1)

        return copy_1
        print(copy_1.molecule.charge)


def get_molecule_list_from_LIBE(
    libe_ids, db, output_folder: str = "outputs"
) -> List[Molecule]:
    """Generate a list of Molecules based on the LIBE IDs."""

    molecule_list = []
    results_collection = db.mp_summary

    for libe_id in libe_ids:
        # Find the molecule entry based on the libe_id
        clean_libe_id = libe_id.replace("libe-", "")
        result = results_collection.find_one({"molecule_id": int(clean_libe_id)})
        molecule = Molecule.from_dict(result["molecule"])
        molecule_list.append(molecule)
        # Write the molecule to a file
        molecule.to(filename=os.path.join(output_folder, f"{clean_libe_id}.xyz"))

    return molecule_list


def get_molecule_from_json(libe_ids: List[str], json_filename) -> List[Molecule]:
    """Get the molecule list from the json file."""

    molecule_list = []

    # Read json file
    with open(json_filename, "r") as f:
        data_dict = json.load(f)

    for data in data_dict:
        if data["molecule_id"] in libe_ids:
            molecule = Molecule.from_dict(data["molecule"])
            molecule_list.append(molecule)
            # Write the molecule to a file
            molecule.to(filename=os.path.join("outputs", f"{data['molecule_id']}.xyz"))
            logging.debug(f"Added molecule {molecule} to the to-fragment list")

    return molecule_list

def get_all_molecule_from_json(json_filename) -> List[Molecule]:
    """Get the molecule list from the json file."""

    molecule_list = []

    # Read json file
    with open(json_filename, "r") as f:
        data_dict = json.load(f)

    for data in data_dict:
        molecule = Molecule.from_dict(data["molecule"])
        molecule_list.append(molecule)
        # Write the molecule to a file
        molecule.to(filename=os.path.join("outputs", f"{data['molecule_id']}.xyz"))
        logging.debug(f"Added molecule {molecule} to the to-fragment list")

    return molecule_list


def get_molecule_graph_from_json(libe_ids: List[str], json_filename) -> List[Molecule]:
    """Get the molecule graph list from the json file."""

    molecule_list = []

    # Read json file
    data_dict = loadfn(json_filename)

    for data in data_dict:
        if data["molecule_id"] in libe_ids:
            molecule_graph = data["molecule_graph"]
            molecule_list.append(molecule_graph)
            # Write the molecule to a file
            logging.debug(f"Added molecule {molecule_graph} to the to-fragment list")

    return molecule_list


def serialize_molecule_graph(molecule_graph_dict):
    """Remove arrays to save the molecule_graph."""
    graph_dict = molecule_graph_dict["graphs"]
    nodes = graph_dict.pop("nodes")
    # Remove the array from the nodes
    for node in nodes:
        array_xyz = node["coords"]
        list_xyz = array_xyz.tolist()
        node["coords"] = list_xyz
    molecule_graph_dict["graphs"]["nodes"] = nodes

