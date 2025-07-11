import logging
import os

import databroker.queries
import numpy as np
import pandas as pd
import tiled.client
import tiled.server.app
from databroker.mongo_normalized import MongoAdapter, discover_handlers
from event_model import Filler
from rich.progress import Progress
from tiled.client.container import Container

COLUMNS = ("x", "y", "x_eta", "y_eta", "y_eta_iso", "sum_regions", "XIP mode")
dtype_list = [(name, "<f4") for name in COLUMNS] + [("frame", "<i2")]
DTYPE = np.dtype(dtype_list)
MAX_LEN = 4800

logger = logging.getLogger(__name__)


class RaggedHandler:
    """
    Adapted from sixtools.AD_Handler:AreaDetector_HDF5SingleHandler_DataFrame.
    Modified to enable a flexible max length stored in the resource as opposed to the server.

    Handler for hdf5 data stored 1 image per file and returned as a
    Pandas.DataFrame.

    This will work with all hdf5 files that are a mxn arrays and the data is
    'table like' where m is the number of columns and n is the number of rows.

    Parameters
    ----------
    fpath : string
        filepath
    template : string
        filename template string.
    filename : string
        filename
    key : string
        the 'path' inside the file to the data set.
    column_names : list[str]
        The column names of the table
    frame_per_point : float
        the number of frames per point.
    max_len : int, optional
        The maximum number of frames to return for each point. If None, defaults to MAX_LEN.
    """

    def __init__(
        self,
        fpath,
        template,
        filename,
        key="/entry/data/data",
        column_names=None,
        frame_per_point=1,
        max_len=None
    ):
        # I have included defaults for `key` and 'column_names' for back
        # compatibility with existing files at SIX.
        self._path = os.path.join(fpath, "")
        self._fpp = frame_per_point
        self._template = template
        self._filename = filename
        self._key = key
        self._column_names = column_names
        self._max_len = MAX_LEN if max_len is None else max_len

    def _fnames_for_point(self, point_number):
        start = int(point_number * self._fpp)
        stop = int((point_number + 1) * self._fpp)
        for j in range(start, stop):
            yield self._template % (self._path, self._filename, j)

    def __call__(self, point_number):
        dfs = []
        import h5py

        for i, fn in enumerate(self._fnames_for_point(point_number)):
            with h5py.File(fn, "r") as f:
                dataframe = pd.DataFrame(f[self._key][:], columns=self._column_names)
                dataframe["frame"] = i
            dfs.append(dataframe)
        records = pd.concat(dfs).to_records(
            column_dtypes={name: dtype for name, dtype in dtype_list}, index=False
        )
        return np.array(records)

    def get_file_list(self, datum_kwargs):
        ret = []
        for d_kw in datum_kwargs:
            ret.extend(self._fnames_for_point(**d_kw))
        return ret


def measure_max_shape(
        *,
        mds_database,
        asset_database,
        descriptor: dict,
        root_map,
        handler_registry,
        data_keys: list[str] | None = None,
):
    """Fills event documents for specific data_keys for the entire run, and returns the maximum shape
    of the data for each key at each dimension."""

    datum_collection = asset_database["datum"]
    resource_collection = asset_database["resource"]
    cursor = mds_database["event"].find(
        {"descriptor": descriptor["uid"]}, sort=[("time", 1)]
    )
    if data_keys is None:
        data_keys = list(descriptor["data_keys"].keys())

    measured_shapes = {}
    for event in cursor:
        filler = Filler(handler_registry=handler_registry, inplace=False, root_map=root_map)
        event["filled"] = {key: False for key in data_keys}
        filler("descriptor", descriptor)
        resources = set()

        for key in data_keys:
            datum = datum_collection.find_one({"datum_id": event["data"][key]})
            if datum["resource"] not in resources:
                resource = resource_collection.find_one({"uid": datum["resource"]})
                filler("resource", resource)
            filler("datum", datum)
        _, filled_event = filler("event", event)

        for key in data_keys:
            data: np.ndarray = filled_event["data"][key]
            if key in measured_shapes:
                measured_shapes[key] = [
                    max(old_dim, new_dim) for old_dim, new_dim in zip(measured_shapes[key], data.shape)
                    ]
            else:
                measured_shapes[key] = [int(dim) for dim in data.shape]
    return measured_shapes


def patch_descriptor_metadata(tiled_client: Container, max_shapes: dict[str, list[int]]):
    """Patches the metadata of the tiled client with the maximum shapes."""
    for key, shape in max_shapes.items():
        tiled_client.patch_metadata(
            [
                {
                    "op": "replace",
                    "path": f"/descriptors/{index}/data_keys/{key}/shape",
                    "value": shape,
                }
                for index in range(len(tiled_client.metadata["descriptors"]))
            ]
        )


def patch_resource_data(tiled_client: Container, max_shapes: dict[str, list[int]]):
    """Patches the resource metadata with the maximum shapes.
    resource = {...
        "uid": "resource_uid",
        resource_kwargs:{..., "max_len": max_shapes["rixscam_centroids"][0]},
        ...
        }
    """
    pass   # TODO: Implement this function to patch the resource data with the max len for centroids.


def max_shape_fix(
        uri: str,
        *,
        asset_registry_uri: str | None = None,
        query: list[str] = [],
        dry_run: bool = True,
        limit: int | None = None,
        root_map: dict[str, str] | None = None,
):
    handler_registry = discover_handlers()
    handler_registry['AD_HDF5_SINGLE_XIP'] = RaggedHandler
    adapter = MongoAdapter.from_uri(uri, asset_registry_uri=asset_registry_uri)
    mds_database = adapter._metadatastore_db
    asset_database = adapter._asset_registry_db

    for q in query:
        parsed_query = eval(q, vars(databroker.queries))
        adapter = adapter.search(parsed_query)
    logger.info(f"Migrating {adapter}")

    root_map = root_map or getattr(adapter, "root_map", {})
    items = adapter.items()
    if limit:
        logger.info(f"Limited to first {limit} BlueskyRuns only")
        items = items[:limit]

    app = tiled.server.app.build_app(adapter)
    with tiled.client.Context.from_app(app) as context:
        tiled_client = tiled.client.from_context(context)

        with Progress() as progress:
            task = progress.add_task("Migrating...", total=len(items))

            for uid, run in items:
                logger.info(f"Processing run {uid}")

                stream_name = "primary"
                stream = run[stream_name]
                descriptor = stream.metadata()["descriptors"][0]

                max_shapes = measure_max_shape(
                    mds_database=mds_database,
                    asset_database=asset_database,
                    descriptor=descriptor,
                    root_map=root_map,
                    handler_registry=handler_registry,
                    data_keys=["rixscam_centroids"]
                )
                if dry_run:
                    msg = "Dry run"
                else:
                    msg = "Edited"
                    # TODO: Uncomment the following lines when the patch functions are implemented.
                    # patch_descriptor_metadata(tiled_client, max_shapes)
                    # patch_resource_data(tiled_client, max_shapes)

                progress.console.print(f"{msg}: Max shapes for run {uid}: {max_shapes}")
            progress.update(task, advance=1)
