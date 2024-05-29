import datetime
import logging
import os
import re
import urllib
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, List, Literal, TypeVar, Union, overload

PathLike = Union[str, bytes, os.PathLike]

import numpy as np
import pandas as pd
import polars as pl
import pytz
import requests
from polars.interchange.protocol import SupportsInterchange
from polars.type_aliases import IpcCompression, ParquetCompression
from tqdm import tqdm

from ..utils.cdh_utils import from_prpc_datetime


def read_ds_export(
    filename: Union[
        str, SupportsInterchange, Literal["model_data", "predictor_data"], BytesIO
    ],
    path: PathLike = ".",
    verbose: bool = True,
    **reading_opts,
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """Read a Pega dataset export file.
    Can accept either a Pandas DataFrame or one of the following formats:
    - .csv
    - .json
    - .zip (zipped json or CSV)
    - .feather
    - .ipc
    - .parquet

    It automatically infers the default file names for both model data as well as predictor data.
    If you supply either 'model_data' or 'predictor_data' as the 'file' argument, it will search for them.
    If you supply the full name of the file in the 'path' directory, it will import that instead.
    Since pdstools V3.x, returns a Polars LazyFrame. Simply call `.collect()` to get an eager frame.

    Parameters
    ----------
    filename : [pd.DataFrame, pl.DataFrame, str]
        Either a Pandas/Polars DataFrame with the source data (for compatibility),
        or a string, in which case it can either be:
        - The name of the file (if a custom name) or
        - Whether we want to look for 'model_data' or 'predictor_data' in the path folder.
    path : PathLike, default = '.'
        The location of the file
    verbose : bool, default = True
        Whether to print out which file will be imported

    Keyword arguments
    -----------------
    Any:
        Any arguments to plug into the scan_* function from Polars.

    Returns
    -------
    pl.LazyFrame
        The (lazy) dataframe

    Examples:
        >>> df = readDSExport(filename = 'modelData', path = './datamart')
        >>> df = readDSExport(filename = 'ModelSnapshot.json', path = 'data/ADMData')

        >>> df = pd.read_csv('file.csv')
        >>> df = readDSExport(filename = df)

    """

    # If a lazy frame is supplied directly, we just pass it through
    if isinstance(filename, (pl.LazyFrame, pl.DataFrame)):
        logging.debug("Polars frame returned directly")
        return filename

    # Through the DataFrame protocol, we can accept any DataFrame-like objects
    if filename and not isinstance(filename, (str, BytesIO, Path)):
        if hasattr(filename, "__dataframe__"):  # it implements the dataframe protocol
            return pl.from_dataframe(filename)

    # If the data is a BytesIO object, such as an uploaded file
    # in certain webapps, then we can simply return the object
    # as is, while extracting the extension as well.
    if isinstance(filename, BytesIO):
        logging.debug("Filename is of type BytesIO, importing that directly")
        _, extension = os.path.splitext(filename.name)
        return import_file(filename, extension, **reading_opts)

    # If the filename is simply a string, then we first
    # extract the extension of the file, then look for
    # the file in the user's directory.
    if os.path.isfile(os.path.join(str(path), str(filename))):
        logging.debug("File found in directory")
        file = os.path.join(str(path), str(filename))
    else:
        logging.debug("File not found in directory, scanning for latest file")
        file = get_latest_file(path, filename)

    # If we can't find the file locally, we can try
    # if the file's a URL. If it is, we need to wrap
    # the file in a BytesIO object, and read the file
    # fully to disk for pyarrow to read it.
    if file == "Target not found" or file is None:
        logging.debug("Could not find file in directory, checking if URL")

        try:
            response = requests.get(f"{path}/{filename}")
            logging.info(f"Response: {response}")
            if response.status_code == 200:
                logging.debug("File found online, importing and parsing to BytesIO")
                file = f"{path}/{filename}"
                file = BytesIO(urllib.request.urlopen(file).read())
                name, extension = os.path.splitext(filename)

        except Exception as e:
            logging.info(e)
            if verbose:
                print(f"File {filename} not found in dir {path}")
            logging.info(f"File not found: {path}/{filename}")
            return None

    if "extension" not in vars():
        name, extension = os.path.splitext(file)

    # Now we should either have a full path to a file, or a
    # BytesIO wrapper around the file. Polars can read those both.
    return import_file(file, extension, **reading_opts)


def import_file(
    file: Union[PathLike, BytesIO], extension: str, **reading_opts
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """Imports a file using Polars

    Parameters
    ----------
    File: str
        The path to the file, passed directly to the read functions
    extension: str
        The extension of the file, used to determine which function to use

    Returns
    -------
    pl.LazyFrame or pl.DataFrame
        The (imported) DataFrame. If we can scan the file, we return a LazyFrame.
        Sometimes we can't scan the file (like for remote files or zips), in which case
        we return a DataFrame. This is useful for inferring if the data actually resides
        in memory or whether the data has not been read yet.
    """

    if extension == ".zip":
        logging.debug("Zip file found, extracting data.json to BytesIO.")
        file, extension = read_zipped_file(file)
    elif extension == ".gz":
        import gzip

        extension = os.path.splitext(os.path.splitext(file)[0])[1]
        file = BytesIO(gzip.GzipFile(file).read())

    if extension == ".csv":
        csv_opts = dict(
            separator=reading_opts.get("sep", ","),
            infer_schema_length=reading_opts.pop("infer_schema_length", 10000),
            null_values=["", "NA", "N/A", "NULL"],
            dtypes={"PYMODELID": pl.Utf8},
            try_parse_dates=True,
            ignore_errors=reading_opts.get("ignore_errors", False),
        )
        if isinstance(file, BytesIO):
            return pl.read_csv(
                file,
                **csv_opts,
            )
        else:
            return pl.scan_csv(file, **csv_opts)

    elif extension == ".json":
        try:
            if isinstance(file, BytesIO):
                from pyarrow import json

                file = pl.DataFrame(
                    json.read_json(
                        file,
                    )
                )
            else:
                return pl.scan_ndjson(
                    file,
                    infer_schema_length=reading_opts.pop("infer_schema_length", 10000),
                )
        except:  # pragma: no cover
            try:
                return pl.read_json(file)
            except:
                import json

                with open(file) as f:
                    return pl.from_dicts(json.loads(f.read())["pxResults"])

    elif extension == ".parquet":
        return pl.scan_parquet(file)

    elif extension.casefold() in {".feather", ".ipc", ".arrow"}:
        if isinstance(file, BytesIO):
            return pl.read_ipc(file)
        else:
            return pl.scan_ipc(file)

    else:
        raise ValueError(f"Could not import file: {file}, with extension {extension}")
    return pl.LazyFrame()


def read_zipped_file(file: PathLike, verbose: bool = False) -> BytesIO:
    """Read a zipped NDJSON file.
    Reads a dataset export file as exported and downloaded from Pega. The export
    file is formatted as a zipped multi-line JSON file. It reads the file,
    and then returns the file as a BytesIO object.

    Parameters
    ----------
    file : PathLike
        The full path to the file
    verbose : str, default=False
        Whether to print the names of the files within the unzipped file for debugging purposes

    Returns
    -------
    os.BytesIO
        The raw bytes object to pass through to Polars
    """

    def _get_valid_files(files):
        logging.debug(f"Files found: {files}")
        if "data.json" in files:
            return "data.json"
        else:  # pragma: no cover
            file = [file for file in files if file.endswith("/data.json")]
            if 0 < len(file) > 1:
                return None
            else:
                return file[0]

    with zipfile.ZipFile(file, mode="r") as z:
        logging.debug("Opened zip file.")
        file = _get_valid_files(z.namelist())
        logging.debug(f"Opening file {file}")
        if file is not None:
            logging.debug("data.json found.")
            if verbose:
                print(
                    (
                        "Zipped json file found. For faster reading, we recommend",
                        "parsing the files to a format such as arrow or parquet. ",
                        "See example in docs #TODO",
                    )
                )
            with z.open(file) as zippedfile:
                return (BytesIO(zippedfile.read()), ".json")
        else:  # pragma: no cover
            raise FileNotFoundError("Cannot find a 'data.json' file in the zip folder.")


def read_multi_zip(
    files: List[PathLike], zip_type: Literal["gzip"] = "gzip", verbose: bool = True
) -> pl.DataFrame:
    """Reads multiple zipped ndjson files, and concats them to one Polars dataframe.

    Parameters
    ----------
    files : list
        The list of files to concat
    zip_type : Literal['gzip']
        At this point, only 'gzip' is supported
    verbose : bool, default = True
        Whether to print out the progress of the import
    """
    import gzip

    table = []
    if zip_type != "gzip":
        raise NotImplementedError("Only supports gzip for now")
    for file in tqdm(files, desc="Combining files...", disable=not verbose):
        table.append(pl.read_ndjson(gzip.open(file).read()))
    df = pl.concat(table, how="diagonal")
    if verbose:
        print("Combining completed")
    return df


def get_latest_file(
    path: PathLike,
    target: Literal["model_data", "predictor_data", "value_finder"],
    verbose: bool = False,
) -> Union[str, None]:
    """Convenience method to find the latest model snapshot.
    It has a set of default names to search for and finds all files who match it.
    Once it finds all matching files in the directory, it chooses the most recent one.
    Supports [".json", ".csv", ".zip", ".parquet", ".feather", ".ipc"].
    Needs a path to the directory and a target of either 'model_data' or 'predictor_data'.

    Parameters
    ----------
    path : PathLike
        The filepath where the data is stored
    target : str in ['model_data', 'predictor_d`ata', 'value_finder']
        Whether to look for data about the predictive models ('model_Data')
        or the predictor bins ('predictor_data')
    verbose : bool, default = False
        Whether to print all found files before comparing name criteria for debugging purposes

    Returns
    -------
    str
        The most recent file given the file name criteria.
    """
    if target not in {"model_data", "predictor_data", "value_finder"}:
        return "Target not found"

    supported = [".json", ".csv", ".zip", ".parquet", ".feather", ".ipc", ".arrow"]

    files_dir = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    files_dir = [f for f in files_dir if os.path.splitext(f)[-1].lower() in supported]
    if verbose:
        print(files_dir)  # pragma: no cover
    matches = get_matches(files_dir, target)

    if len(matches) == 0:  # pragma: no cover
        if verbose:
            print(
                f"Unable to find data for {target}. Please check if the data is available."
            )
        return None

    paths = [os.path.join(path, name) for name in matches]

    def f(x):
        try:
            return from_prpc_datetime(re.search("\d.*GMT", x)[0].replace("_", " "))
        except Exception:
            return pytz.timezone("GMT").localize(
                datetime.datetime.fromtimestamp(os.path.getctime(x))
            )

    dates = np.array([f(i) for i in paths])
    return paths[np.argmax(dates)]


def get_matches(files_dir, target):
    matches = []
    default_model_names = [
        "Data-Decision-ADM-ModelSnapshot",
        "PR_DATA_DM_ADMMART_MDL_FACT",
        "model_snapshots",
        "MD_FACT",
        "ADMMART_MDL_FACT_Data",
        "cached_model_data",
        "Models_data",
    ]
    default_predictor_names = [
        "Data-Decision-ADM-PredictorBinningSnapshot",
        "PR_DATA_DM_ADMMART_PRED",
        "predictor_binning_snapshots",
        "PRED_FACT",
        "cached_predictor_data",
    ]
    value_finder_names = ["Data-Insights_pyValueFinder", "cached_ValueFinder"]

    if target == "model_data":
        names = default_model_names
    elif target == "predictor_data":
        names = default_predictor_names
    elif target == "ValueFinder":
        names = value_finder_names
    else:
        raise ValueError(f"Target {target} not found.")
    for file in files_dir:
        match = [file for name in names if re.findall(name.casefold(), file.casefold())]
        if len(match) > 0:
            matches.append(match[0])
    return matches


@overload
def cache_to_file(
    df: Union[pl.DataFrame, pl.LazyFrame],
    path: PathLike,
    name: str,
    cache_type: Literal["ipc"] = "ipc",
    compression: IpcCompression = "uncompressed",
) -> Path:
    ...


@overload
def cache_to_file(
    df: Union[pl.DataFrame, pl.LazyFrame],
    path: PathLike,
    name: str,
    cache_type: Literal["parquet"] = "parquet",
    compression: ParquetCompression = "uncompressed",
) -> Path:
    ...


def cache_to_file(
    df: Union[pl.DataFrame, pl.LazyFrame],
    path: PathLike,
    name: str,
    cache_type: Literal["ipc", "parquet"] = "ipc",
    compression: Union[ParquetCompression, IpcCompression] = "uncompressed",
) -> Path:
    """Very simple convenience function to cache data.
    Caches in arrow format for very fast reading.

    Parameters
    ----------
    df: pl.DataFrame
        The dataframe to cache
    path: os.PathLike
        The location to cache the data
    name: str
        The name to give to the file
    cache_type: str
        The type of file to export.
        Default is IPC, also supports parquet
    compression: str
        The compression to apply, default is uncompressed

    Returns
    -------
    os.PathLike:
        The filepath to the cached file
    """
    import pathlib

    outpath = pathlib.Path(path).joinpath(pathlib.Path(name))
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    if cache_type == "ipc":
        outpath = outpath.joinpath(".arrow")
        df.write_ipc(outpath, compression=compression)
    if cache_type == "parquet":
        outpath = outpath.joinpath(".parquet")
        df.write_parquet(outpath, compression=compression)
    return outpath
