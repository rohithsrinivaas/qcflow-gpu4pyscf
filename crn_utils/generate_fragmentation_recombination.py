import argparse

from instance_mongodb import instance_mongodb_sei

from fragmentation_recombination import get_molecule_from_json, FragmentReconnect


def parse_cli():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="The depth of the fragmentation.",
    )
    parser.add_argument(
        "--libe_ids",
        type=str,
        nargs="+",
        #default=["mvbe-980419", "mvbe-855003", "mvbe-320736", "mvbe-877247", "mvbe-487010", "mvbe-100696", "mvbe-306714"],
        default=["mvbe-980419", "mvbe-877247", "mvbe-487010", "mvbe-100696", "mvbe-306714", "mvbe-299570", "mvbe-531016", "mvbe-369576", "mvbe-826469"],
        help="The LIBE IDs of the molecules to fragment.",
    )
    parser.add_argument(
        "--json_filename",
        type=str,
        default="mg_initial_structures_12_06_2023.json",
        help="The filename of the JSON file containing the LIBE IDs.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """Fragment a molecule."""

    # Get the database and create a collection to store the structures in
    db = instance_mongodb_sei()

    # Get the CLI arguments
    args = parse_cli()

    # Create a collection to store the fragments
    initial_graphs_collection = db.Mg_cluster_initial_structures_12_06_2023

    # Generate a molecule list
    molecule_list = get_molecule_from_json(args.libe_ids, args.json_filename)

    fragmenter = FragmentReconnect(
        initial_graphs_collection=initial_graphs_collection,
        molecule_list=molecule_list,
        depth=args.depth,
        bonding_factor_max=1.5,
        bonding_factor_min=1,
        bonding_factor_number=3,
        number_of_angles=100,
        debug=args.debug,
    )
    fragmenter()
