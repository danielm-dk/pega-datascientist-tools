"""Python pdstools"""

__version__ = "V4.0.0"

from polars import enable_string_cache

enable_string_cache()

import sys
from pathlib import Path

from .adm.ADMDatamart import ADMDatamart
from .adm.ADMTrees import ADMTrees, MultiTrees
from .adm.BinAggregator import BinAggregator
from .pega_io import API, S3, File, get_token, read_ds_export
from .pega_io.API import setupAzureOpenAI
from .prediction import Prediction
from .utils import NBAD, cdh_utils, datasets, errors, hds_utils
from .utils.cdh_utils import default_predictor_categorization
from .utils.CDHLimits import CDHLimits
from .utils.datasets import CDHSample, SampleTrees, SampleValueFinder
from .utils.hds_utils import Config, DataAnonymization
from .utils.polars_ext import *
from .utils.show_versions import show_versions
from .utils.table_definitions import PegaDefaultTables
from .valuefinder.ValueFinder import ValueFinder

if "streamlit" in sys.modules:
    from .utils import streamlit_utils

__reports__ = Path(__file__).parents[0] / "reports"
