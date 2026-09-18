"""Flat-file replacement for MongoDB storage.

`JsonFileCollection` implements just the two MongoDB collection calls actually
used elsewhere in this package (`FragmentReconnect._get_graphs_already_in_database`
and `._put_graph_in_database`/`._put_monoatomic_graph_in_database`, in
`fragmentation_recombination.py`): `.find({"tags.<key>": value})` and
`.insert_one(doc)`. `write_result`/`iter_results` are the equivalent for the
per-molecule GPU4PySCF result documents written by the driver scripts.
"""

import glob
import json
import os
import uuid


def write_result(directory, name, doc):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.json")
    with open(path, "w") as f:
        json.dump(doc, f)
    return path


def iter_results(directory):
    if not os.path.isdir(directory):
        return
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        with open(path) as f:
            yield json.load(f)


def result_exists(directory, name):
    return os.path.exists(os.path.join(directory, f"{name}.json"))


class JsonFileCollection:
    """One JSON file per document in `directory`, duck-typing the subset of a
    MongoDB collection that FragmentReconnect relies on.
    """

    def __init__(self, directory):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def find(self, query):
        for key, value in query.items():
            if not key.startswith("tags."):
                raise NotImplementedError(f"Unsupported query key: {key}")
        tag_keys = {key[len("tags."):]: value for key, value in query.items()}

        for doc in iter_results(self.directory):
            tags = doc.get("tags", {})
            if all(tags.get(k) == v for k, v in tag_keys.items()):
                yield doc

    def insert_one(self, doc):
        doc_id = uuid.uuid4().hex
        doc["_id"] = doc_id
        write_result(self.directory, doc_id, doc)
