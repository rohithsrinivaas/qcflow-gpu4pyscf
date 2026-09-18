from crn_utils.fragmentation_recombination import get_all_molecule_from_json, FragmentReconnect
from crn_utils.file_store import JsonFileCollection
import os
import yaml


# Get the config
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Create a flat-file collection to store the fragments (replaces the MongoDB
# collection previously named "<group>_initial_graphs_collection")
initial_graphs_collection = JsonFileCollection(
    os.path.join("outputs", "initial_graphs")
)


# Generate a molecule list
molecule_list = get_all_molecule_from_json('init_optimized.json')


fragmenter = FragmentReconnect(
    initial_graphs_collection=initial_graphs_collection,
    molecule_list=molecule_list,
    depth= 2,
    bonding_factor_max=1.5,
    bonding_factor_min=1,
    bonding_factor_number=3,
    number_of_angles=100,
    debug= False, # args.debug,
)

fragmenter.run(if_add_monoatomic = False)
