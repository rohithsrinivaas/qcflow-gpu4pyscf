import logging

from monty.serialization import dumpfn
from pymatgen.core.structure import Molecule

from crn_utils.utils import molecule_barcode, molecule_id_from_barcode
from crn_utils.file_store import iter_results


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig()


current_data = []
seen_barcodes = {}

for idx, result in enumerate(iter_results("outputs/init_optimized")):
    if result.get("error"):
        logger.warning(f"Skipping {idx} because it errored: {result['error']}")
        continue

    molecule = Molecule.from_dict(result["molecule"])
    barcode = molecule_barcode(molecule)
    if "Error in MolBar generation" in barcode:
        logger.warning(f"Skipping {idx} because MolBar failed: {barcode}")
        continue

    if barcode in seen_barcodes:
        logger.warning(f"Skipping {idx} because it duplicates {seen_barcodes[barcode]}")
        continue

    molecule_id = molecule_id_from_barcode(barcode)
    seen_barcodes[barcode] = molecule_id
    logger.info(f"Adding {idx} with id {molecule_id}")
    current_data.append({
        "molecule_id": molecule_id,
        "molbar": barcode,
        "molecule": result["molecule"],
    })


new_file_name = "init_optimized.json"
print(f"Writing {len(current_data)} molecules to {new_file_name}")
dumpfn(current_data, new_file_name)
