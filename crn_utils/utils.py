import hashlib
import json
import os

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN

from mrnet.core.mol_entry import MoleculeEntry, MoleculeEntryError
from mrnet.core.reactions import bucket_mol_entries
from molbar.barcode import get_molbar_from_coordinates

from typing import List, Tuple


def mkdir(path: str):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print("Folder exists")
    return path


def get_all_graphs(data):
    """Extract all graphs from the data."""
    all_graphs = []
    all_ids = []
    for entry in data:
        all_graphs.append(entry["molecule_graph"])
        all_ids.append(entry["molecule_id"])
    return all_graphs, all_ids


def get_all_graphs_from_collection(collection):
    """Extract all graphs from the data (flat GPU4PySCF result documents)."""
    all_graphs = []
    for entry in collection:
        molecule = Molecule.from_dict(entry["initial_molecule"])
        # Create the molecule graph
        molecule_graph = MoleculeGraph.from_local_env_strategy(molecule, OpenBabelNN())
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


def _dedup_key(molecule_graph):
    """Cheap invariant that isomorphic graphs must share: (formula, charge,
    spin). Used to bucket a set of graphs so a duplicate check only compares
    against same-formula/charge/spin candidates instead of every graph in the
    set -- a full linear isomorphism scan against a large collection (tens of
    thousands of graphs) is prohibitively slow otherwise.
    """
    mol = molecule_graph.molecule
    return (mol.composition.formula, mol.charge, mol.spin_multiplicity)


def build_dedup_index(graphs):
    """Bucket `graphs` (an iterable of MoleculeGraph) by _dedup_key for fast
    is_duplicate() lookups."""
    index = {}
    for graph in graphs:
        index.setdefault(_dedup_key(graph), []).append(graph)
    return index


def is_duplicate(molecule_graph, index):
    """Check molecule_graph against only the matching bucket of a
    build_dedup_index() result, instead of a full linear scan."""
    for graph in index.get(_dedup_key(molecule_graph), []):
        if molecule_graph.isomorphic_to(graph):
            return True
    return False


def graphs_from_json_collection(directory):
    """Yield MoleculeGraph objects stored in a JsonFileCollection directory
    (as written by FragmentReconnect / JsonFileCollection.insert_one), e.g.
    another molecule's outputs/initial_graphs directory, for cross-run dedup.
    """
    from pymatgen.analysis.graphs import MoleculeGraph as _MoleculeGraph
    from crn_utils.file_store import iter_results

    for doc in iter_results(directory):
        doc = dict(doc)
        doc.pop("tags", None)
        doc.pop("_id", None)
        yield _MoleculeGraph.from_dict(doc)



def molecule_barcode(molecule: Molecule) -> str:
    """MolBar structural identifier for `molecule`, computed from its 3D
    geometry, element list, and total charge. Deterministic across runs
    (small geometry noise doesn't change it) and distinguishes charge and
    stereochemistry, so identical barcodes mean identical species."""
    elements = [str(s) for s in molecule.species]
    coordinates = molecule.cart_coords.tolist()
    return get_molbar_from_coordinates(coordinates, elements, total_charge=int(molecule.charge))


def molecule_id_from_barcode(barcode: str) -> str:
    """Short filesystem-safe id derived from a MolBar barcode."""
    digest = hashlib.sha256(barcode.encode()).hexdigest()[:12]
    return f"mol-{digest}"


def get_external_duplicate_ids(initial_graphs_dir, external_dirs, cache_path):
    """Return the set of doc_ids in `initial_graphs_dir` that are isomorphic
    (same formula/charge/spin and graph) to some structure in one of
    `external_dirs` (other JsonFileCollection directories, e.g. another
    molecule's outputs/initial_graphs) -- so a downstream step can skip
    optimizing structures another run already covers.

    Cached to `cache_path` with no automatic invalidation: rebuilding
    requires loading every external graph and scanning every candidate
    against them, expensive enough (tens of thousands of graphs) that redoing
    it on every invocation of a script meant to be launched repeatedly (e.g.
    across many walltime-limited GPU jobs) would be wasteful. A file-count
    based staleness check was tried and dropped -- on this parallel
    filesystem, `os.listdir()` counts on a directory with tens of thousands
    of files can keep drifting for a while after a big concurrent write burst
    (metadata listing catching up, no new content), which falsely
    invalidated the cache every time. Since nothing downstream of
    fragmentation/recombination ever writes back into `initial_graphs_dir`,
    the safe assumption is a built cache stays valid until the candidate pool
    is deliberately changed (e.g. script 3 rerun/extended) -- in that case,
    delete `cache_path` to force a rebuild.
    """
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        return set(cached["duplicate_ids"])

    from crn_utils.file_store import iter_results

    external_index = build_dedup_index(
        graph for d in external_dirs for graph in graphs_from_json_collection(d)
    )

    duplicate_ids = set()
    for doc in iter_results(initial_graphs_dir):
        doc = dict(doc)
        doc_id = doc.pop("_id")
        doc.pop("tags", None)
        molecule_graph = MoleculeGraph.from_dict(doc)
        if is_duplicate(molecule_graph, external_index):
            duplicate_ids.add(doc_id)

    with open(cache_path, "w") as f:
        json.dump({"duplicate_ids": sorted(duplicate_ids)}, f)

    return duplicate_ids
