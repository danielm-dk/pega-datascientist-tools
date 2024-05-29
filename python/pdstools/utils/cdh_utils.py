# -*- coding: utf-8 -*-
"""
cdhtools: Data Science add-ons for Pega.

Various utilities to access and manipulate data from Pega for purposes
of data analysis, reporting and monitoring.
"""
import datetime
import re
from typing import Iterable, List, Literal, Optional, TypeVar, Union, overload

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytz

from .errors import NotEagerError

frame = TypeVar("frame", pl.DataFrame, pl.LazyFrame)


def default_predictor_categorization(
    x: Union[str, pl.Expr] = pl.col("PredictorName"),
) -> pl.Expr:
    """Function to determine the 'category' of a predictor.

    It is possible to supply a custom function.
    This function can accept an optional column as input
    And as output should be a Polars expression.
    The most straight-forward way to implement this is with
    pl.when().then().otherwise(), which you can chain.

    By default, this function returns "Primary" whenever
    there is no '.' anywhere in the name string,
    otherwise returns the first string before the first period

    Parameters
    ----------
    x: Union[str, pl.Expr], default = pl.col('PredictorName')
        The column to parse

    """
    if isinstance(x, str):
        x = pl.col(x)
    x = x.cast(pl.Utf8) if not isinstance(x, pl.Utf8) else x
    return (
        pl.when(x.str.split(".").list.len() > 1)
        .then(x.str.split(".").list.get(0))
        .otherwise(pl.lit("Primary"))
    ).alias("PredictorCategory")


def _extract_keys(
    df: frame,
    col="Name",
    capitalize=True,
    import_strategy="eager",
) -> pl.Expr:
    """Extracts keys out of the pyName column

    This is not a lazy operation as we don't know the possible keys
    in advance. For that reason, we select only the pyName column,
    extract the keys from that, and then collect the resulting dataframe.
    This dataframe is then joined back to the original dataframe.

    This is relatively efficient, but we still do need the whole
    pyName column in memory to do this, so it won't work completely
    lazily from e.g. s3. That's why it only works with eager mode.

    Parameters
    ----------
    df: Union[pl.DataFrame, pl.LazyFrame]
        The dataframe to extract the keys from
    """
    if import_strategy != "eager":
        raise NotEagerError("Extracting keys")

    def safe_name():
        return (
            pl.when(~pl.col(col).cast(pl.Utf8).str.starts_with("{"))
            .then(pl.concat_str([pl.lit('{"pyName":"'), pl.col(col), pl.lit('"}')]))
            .otherwise(pl.col(col))
        ).alias("tempName")

    series = (
        df.select(
            safe_name().str.json_decode(),
        )
        .unnest("tempName")
        .lazy()
        .collect()
    )
    if not capitalize:
        return df.with_columns(series)
    return _polars_capitalize(series)


def parse_pega_datetime_formats(
    timestamp_col="SnapshotTime",
    timestamp_fmt: Optional[str] = None,
    strict_conversion: bool = True,
) -> pl.Expr:
    """Parses Pega DateTime formats.

    Supports the two most commonly used formats:

    - "%Y-%m-%d %H:%M:%S"
    - "%Y%m%dT%H%M%S.%f %Z"

    If you want to parse a different timezone, then

    Removes timezones, and rounds to seconds, with a 'ns' time unit.

    Parameters
    ----------
    timestampCol: str, default = 'SnapshotTime'
        The column to parse
    timestamp_fmt: str, default = None
        An optional format to use rather than the default formats
    strict_conversion: bool, default = True
        Whether to error on incorrect parses or just return Null values
    """
    if timestamp_fmt is not None:
        return pl.col(timestamp_col).str.strptime(
            pl.Datetime,
            timestamp_fmt,
            strict=strict_conversion,
        )
    else:
        return (
            pl.when((pl.col(timestamp_col).str.slice(4, 1) == pl.lit("-")))
            .then(
                pl.col(timestamp_col)
                .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
                .dt.cast_time_unit("ns")
            )
            .otherwise(
                pl.col(timestamp_col)
                .str.strptime(pl.Datetime, "%Y%m%dT%H%M%S.%3f %Z", strict=False)
                .dt.replace_time_zone(None)
                .dt.cast_time_unit("ns")
            )
        ).alias(timestamp_col)


# TODO Commenting this out for now while doing refactoring
# This code should be removed because we should have a better solution for it
# We should just maintain proper table dictionaries where we have this info available

# def get_type_mapping(df, definition, verbose:bool=False, **timestamp_opts):
#     """
#     This function is used to convert the data types of columns in a DataFrame to a desired types.
#     The desired types are defined in a `PegaDefaultTables` class.

#     Parameters
#     ----------
#     df : pl.LazyFrame
#         The DataFrame whose columns' data types need to be converted.
#     definition : PegaDefaultTables
#         A `PegaDefaultTables` object that contains the desired data types for the columns.
#     verbose : bool
#         If True, the function will print a message when a column is not in the default table schema.
#     timestamp_opts : str
#         Additional arguments for timestamp parsing.

#     Returns
#     -------
#     List
#         A list with polars expressions for casting data types.
#     """

#     def getMapping(columns, reverse=False):
#         if not reverse:
#             return dict(zip(columns, _capitalize(columns)))
#         else:
#             return dict(zip(_capitalize(columns), columns))

#     named = getMapping(df.columns)
#     typed = getMapping(
#         [col for col in dir(definition) if not col.startswith("__")], reverse=True
#     )

#     types = []
#     for col, renamedCol in named.items():
#         try:
#             new_type = getattr(definition, typed[renamedCol])
#             original_type = df.schema[col].base_type()
#             if original_type == pl.Null:
#                 if verbose:
#                     warnings.warn(f"Warning: {col} column is Null data type.")
#             elif original_type != new_type:
#                 if original_type == pl.Categorical and new_type in pl.NUMERIC_DTYPES:
#                     types.append(pl.col(col).cast(pl.Utf8).cast(new_type))
#                 elif new_type == pl.Datetime and original_type != pl.Date:
#                     types.append(parse_pega_datetime_formats(col, **timestamp_opts))
#                 else:
#                     types.append(pl.col(col).cast(new_type))
#         except:
#             if verbose:
#                 warnings.warn(
#                     f"Column {col} not in default table schema, can't set type."
#                 )
#     return types


# def set_types(df, table="infer", verbose=False, **timestamp_opts):
#     if table == "infer":
#         table = inferTableDefinition(df)

#     if table == "pyValueFinder":
#         definition = PegaDefaultTables.pyValueFinder()
#     elif table == "ADMModelSnapshot":
#         definition = PegaDefaultTables.ADMModelSnapshot()
#     elif table == "ADMPredictorBinningSnapshot":
#         definition = PegaDefaultTables.ADMPredictorBinningSnapshot()

#     else:
#         raise ValueError(table)

#     mapping = get_type_mapping(df, definition, verbose=verbose, **timestamp_opts)

#     if len(mapping) > 0:
#         return df.with_columns(mapping)
#     else:
#         return df


# def inferTableDefinition(df):
#     cols = _capitalize(df.columns)
#     vf = ["Propensity", "Stage"]
#     predictors = ["PredictorName", "ModelID", "BinSymbol"]
#     models = ["ModelID", "Performance"]
#     if all(value in cols for value in vf):
#         return "pyValueFinder"
#     elif all(value in cols for value in predictors):
#         return "ADMPredictorBinningSnapshot"
#     elif all(value in cols for value in models):
#         return "ADMModelSnapshot"
#     else:
#         print("Could not find table definition.")
#         return cols


def safe_range_auc(auc: float) -> float:
    """Internal helper to keep auc a safe number between 0.5 and 1.0 always.

    Parameters
    ----------
    auc : float
        The AUC (Area Under the Curve) score

    Returns
    -------
    float
        'Safe' AUC score, between 0.5 and 1.0
    """

    if np.isnan(auc):
        return 0.5
    else:
        return 0.5 + np.abs(0.5 - auc)


def auc_from_probs(
    groundtruth: List[int], probs: List[float]
) -> float:  # pragma: no cover
    """Calculates AUC from an array of truth values and predictions.
    Calculates the area under the ROC curve from an array of truth values and
    predictions, making sure to always return a value between 0.5 and 1.0 and
    returns 0.5 when there is just one groundtruth label.

    Parameters
    ----------
    groundtruth : List[int]
        The 'true' values, Positive values must be represented as
        True or 1. Negative values must be represented as False or 0.
    probs : List[float]
        The predictions, as a numeric vector of the same length as groundtruth

    Returns : float
        The AUC as a value between 0.5 and 1.

    Examples:
        >>> auc_from_probs( [1,1,0], [0.6,0.2,0.2])
    """
    nlabels = len(np.unique(groundtruth))
    if nlabels < 2:
        return 0.5
    if nlabels > 2:
        raise Exception("'Groundtruth' has more than two levels.")

    df = pl.DataFrame({"truth": groundtruth, "probs": probs})
    binned = df.group_by(by="probs").agg(
        [
            (pl.col("truth") == 1).sum().alias("pos"),
            (pl.col("truth") == 0).sum().alias("neg"),
        ]
    )

    return auc_from_bincounts(
        binned.get_column("pos"), binned.get_column("neg"), binned.get_column("probs")
    )


def auc_from_bincounts(
    pos: Iterable[int], neg: Iterable[int], probs: Optional[Iterable[float]] = None
) -> float:
    """Calculates AUC from counts of positives and negatives directly
    This is an efficient calculation of the area under the ROC curve directly from an array of positives
    and negatives. It makes sure to always return a value between 0.5 and 1.0
    and will return 0.5 when there is just one groundtruth label.

    Parameters
    ----------
    pos : List[int]
        Vector with counts of the positive responses
    neg: List[int]
        Vector with counts of the negative responses
    probs: List[float]
        Optional list with probabilities which will be used to set the order of the bins. If missing defaults to pos/(pos+neg).

    Returns
    -------
    float
        The AUC as a value between 0.5 and 1.

    Examples:
        >>> auc_from_bincounts([3,1,0], [2,0,1])
    """
    pos_arr = np.asarray(pos, dtype=np.uint32)
    neg_arr = np.asarray(neg, dtype=np.uint32)
    if probs is None:
        probs = np.asarray(pos_arr / (pos_arr + neg_arr), dtype=np.float32)

    binorder = np.argsort(probs)[::-1]
    FPR = np.cumsum(neg_arr[binorder]) / np.sum(neg_arr)
    TPR = np.cumsum(pos_arr[binorder]) / np.sum(pos_arr)

    area = (np.diff(FPR, prepend=0)) * (TPR + np.insert(np.roll(TPR, 1)[1:], 0, 0)) / 2
    return safe_range_auc(np.sum(area))


def aucpr_from_probs(
    groundtruth: List[int], probs: List[float]
) -> float:  # pragma: no cover
    """Calculates PR AUC (precision-recall) from an array of truth values and predictions.
    Calculates the area under the PR curve from an array of truth values and
    predictions. Returns 0.0 when there is just one groundtruth label.

    Parameters
    ----------
    groundtruth : List[int]
        The 'true' values, Positive values must be represented as
        True or 1. Negative values must be represented as False or 0.
    probs : List[float]
        The predictions, as a numeric vector of the same length as groundtruth

    Returns : float
        The AUC as a value between 0.5 and 1.

    Examples:
        >>> auc_from_probs( [1,1,0], [0.6,0.2,0.2])
    """
    nlabels = len(np.unique(groundtruth))
    if nlabels < 2:
        return 0.0  # Again, is this right? Shouldn't we return a list here?
    if nlabels > 2:
        raise Exception("'Groundtruth' has more than two levels.")

    df = pl.DataFrame({"truth": groundtruth, "probs": probs})
    binned = df.group_by(by="probs").agg(
        [
            (pl.col("truth") == 1).sum().alias("pos"),
            (pl.col("truth") == 0).sum().alias("neg"),
        ]
    )

    return aucpr_from_bincounts(
        binned.get_column("pos"), binned.get_column("neg"), binned.get_column("probs")
    )


def aucpr_from_bincounts(
    pos: Iterable[int], neg: Iterable[int], probs: Optional[Iterable[float]] = None
) -> float:
    """Calculates PR AUC (precision-recall) from counts of positives and negatives directly.
    This is an efficient calculation of the area under the PR curve directly from an
    array of positives and negatives. Returns 0.0 when there is just one
    groundtruth label.

    Parameters
    ----------
    pos : List[int]
        Vector with counts of the positive responses
    neg: List[int]
        Vector with counts of the negative responses
    probs: List[float]
        Optional list with probabilities which will be used to set the order of the bins. If missing defaults to pos/(pos+neg).

    Returns
    -------
    float
        The PR AUC as a value between 0.0 and 1.

    Examples:
        >>> aucpr_from_bincounts([3,1,0], [2,0,1])
    """
    pos_arr = np.asarray(pos, dtype=np.uint32)
    neg_arr = np.asarray(neg, dtype=np.uint32)
    if probs is None:
        o = np.argsort(-(pos_arr / (pos_arr + neg_arr)))
    else:
        o = np.argsort(-np.asarray(probs))
    recall = np.cumsum(pos_arr[o]) / np.sum(pos_arr)
    precision = np.cumsum(pos_arr[o]) / np.cumsum(pos_arr[o] + neg_arr[o])
    prevrecall = np.insert(recall[0 : (len(recall) - 1)], 0, 0)
    prevprecision = np.insert(precision[0 : (len(precision) - 1)], 0, 0)
    area = (recall - prevrecall) * (precision + prevprecision) / 2
    return np.sum(area[1:])


def auc_to_gini(auc: float) -> float:
    """
    Convert AUC performance metric to GINI

    Parameters
    ----------
    auc: float
        The AUC (number between 0.5 and 1)

    Returns
    -------
    float
        GINI metric, a number between 0 and 1

    Examples:
        >>> auc_to_gini(0.8232)
    """
    return 2 * safe_range_auc(auc) - 1


def _capitalize(fields: Iterable[str]) -> List[str]:
    """Applies automatic capitalization, aligned with the R couterpart.

    Parameters
    ----------
    fields : list
        A list of names

    Returns
    -------
    fields : list
        The input list, but each value properly capitalized
    """
    capitalize_endwords = [
        "ID",
        "Key",
        "Name",
        "Treatment",
        "Count",
        "Category",
        "Class",
        "Time",
        "DateTime",
        "UpdateTime",
        "ToClass",
        "Version",
        "Predictor",
        "Predictors",
        "Rate",
        "Ratio",
        "Negatives",
        "Positives",
        "Threshold",
        "Error",
        "Importance",
        "Type",
        "Percentage",
        "Index",
        "Symbol",
        "LowerBound",
        "UpperBound",
        "Bins",
        "GroupIndex",
        "ResponseCount",
        "NegativesPercentage",
        "PositivesPercentage",
        "BinPositives",
        "BinNegatives",
        "BinResponseCount",
        "BinSymbol",
        "ResponseCountPercentage",
        "ConfigurationName",
        "Configuration",
    ]
    if not isinstance(fields, list):
        fields = [fields]
    fields = [re.sub("^p(x|y|z)", "", field.lower()) for field in fields]
    fields = list(
        map(lambda x: x.replace("configurationname", "configuration"), fields)
    )
    for word in capitalize_endwords:
        fields = [re.sub(word, word, field, flags=re.I) for field in fields]
        fields = [field[:1].upper() + field[1:] for field in fields]
    return fields


def _polars_capitalize(df: frame) -> frame:
    return df.rename(
        dict(
            zip(
                df.columns,
                _capitalize(df.columns),
            )
        )
    )


@overload
def from_prpc_datetime(
    x: str, return_string: Literal[False] = False
) -> datetime.datetime:
    ...


@overload
def from_prpc_datetime(x: str, return_string: Literal[True]) -> str:
    ...


def from_prpc_datetime(
    x: str, return_string: bool = False
) -> Union[datetime.datetime, str]:
    """Convert from a Pega date-time string.

    Parameters
    ----------
    x: str
        String of Pega date-time
    return_string: bool, default=False
        If True it will return the date in string format. If
        False it will return in datetime type

    Returns
    -------
    Union[datetime.datetime, str]
        The converted date in datetime format or string.

    Examples:
        >>> fromPRPCDateTime("20180316T134127.847 GMT")
        >>> fromPRPCDateTime("20180316T134127.847 GMT", True)
        >>> fromPRPCDateTime("20180316T184127.846")
        >>> fromPRPCDateTime("20180316T184127.846", True)
    """

    timezonesplits = x.split(" ")

    if len(timezonesplits) > 1:
        x = timezonesplits[0]

    if "." in x:
        date_no_frac, frac_sec = x.split(".")
        # TODO: obtain only 3 decimals
        if len(frac_sec) > 3:
            frac_sec = frac_sec[:3]
        elif len(frac_sec) < 3:
            frac_sec = "{:<03d}".format(int(frac_sec))
    else:
        date_no_frac = x

    dt = datetime.datetime.strptime(date_no_frac, "%Y%m%dT%H%M%S")

    if len(timezonesplits) > 1:
        dt = dt.replace(tzinfo=pytz.timezone(timezonesplits[1]))

    if "." in x:
        dt = dt.replace(microsecond=int(frac_sec))

    if return_string:
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    else:
        return dt


def to_prpc_datetime(dt: datetime.datetime) -> str:
    """Convert to a Pega date-time string

    Parameters
    ----------
    x: datetime.datetime
        A datetime object

    Returns
    -------
    str
        A string representation in the format used by Pega

    Examples:
        >>> toPRPCDateTime(datetime.datetime.now())
    """
    if dt.tzinfo is None:
        dt = dt.astimezone()
    return dt.strftime("%Y%m%dT%H%M%S.%f")[:-3] + dt.strftime(" GMT%z")


def weighted_average_polars(
    vals: Union[str, pl.Expr], weights: Union[str, pl.Expr]
) -> pl.Expr:
    if isinstance(vals, str):
        vals = pl.col(vals)
    if isinstance(weights, str):
        weights = pl.col(weights)

    return (
        (vals * weights).filter(vals.is_not_nan() & weights.is_not_null()).sum()
    ) / weights.sum()


def weighted_performance_polars() -> pl.Expr:
    """Polars function to return a weighted performance"""
    return weighted_average_polars("Performance", "ResponseCount")


def z_ratio(
    pos_col: pl.Expr = pl.col("BinPositives"), neg_col: pl.Expr = pl.col("BinNegatives")
) -> pl.Expr:
    """Calculates the Z-Ratio for predictor bins.

    The Z-ratio is a measure of how the propensity in a bin differs from the average,
    but takes into account the size of the bin and thus is statistically more relevant.
    It represents the number of standard deviations from the avreage,
    so centers around 0. The wider the spread, the better the predictor is.

    To recreate the OOTB ZRatios from the datamart, use in a group_by.
    See `examples`.

    Parameters
    ----------
    pos_col: pl.Expr
        The (Polars) column of the bin positives
    neg_col: pl.Expr
        The (Polars) column of the bin positives

    Examples
    --------
    >>> df.group_by(['ModelID', 'PredictorName']).agg([z_ratio()]).explode()
    """

    def get_fracs(pos_col=pl.col("BinPositives"), neg_col=pl.col("BinNegatives")):
        return pos_col / pos_col.sum(), neg_col / neg_col.sum()

    def z_ratio_impl(
        pos_fraction_col=pl.col("posFraction"),
        neg_fraction_col=pl.col("negFraction"),
        positives_col=pl.sum("BinPositives"),
        negatives_col=pl.sum("BinNegatives"),
    ):
        return (
            (pos_fraction_col - neg_fraction_col)
            / (
                (pos_fraction_col * (1 - pos_fraction_col) / positives_col)
                + (neg_fraction_col * (1 - neg_fraction_col) / negatives_col)
            ).sqrt()
        ).alias("ZRatio")

    return z_ratio_impl(*get_fracs(pos_col, neg_col), pos_col.sum(), neg_col.sum())


def lift(
    pos_col: pl.Expr = pl.col("BinPositives"), neg_col: pl.Expr = pl.col("BinNegatives")
) -> pl.Expr:
    """Calculates the Lift for predictor bins.

    The Lift is the ratio of the propensity in a particular bin over the average
    propensity. So a value of 1 is the average, larger than 1 means higher
    propensity, smaller means lower propensity.

    Parameters
    ----------
    posCol: pl.Expr
        The (Polars) column of the bin positives
    negCol: pl.Expr
        The (Polars) column of the bin positives

    Examples
    --------
    >>> df.group_by(['ModelID', 'PredictorName']).agg([lift()]).explode()
    """

    def lift_impl(bin_pos, bin_neg, total_pos, total_neg):
        return (
            # TODO not sure how polars (mis)behaves when there are no positives at all
            # I would hope for a NaN but base python doesn't do that. Polars perhaps.
            # Stijn: It does have proper None value support, may work like you say
            bin_pos * (total_pos + total_neg) / ((bin_pos + bin_neg) * total_pos)
        ).alias("Lift")

    return lift_impl(pos_col, neg_col, pos_col.sum(), neg_col.sum())


def log_odds(
    positives=pl.col("Positives"),
    negatives=pl.col("ResponseCount") - pl.col("Positives"),
) -> pl.Expr:
    N = positives.count()
    return (
        (
            ((positives + 1 / N).log() - (positives + 1).sum().log())
            - ((negatives + 1 / N).log() - (negatives + 1).sum().log())
        )
        .round(2)
        .alias("LogOdds")
    )


# TODO: reconsider this. Feature importance now stored in datamart
# perhaps we should not bother to calculate it ourselves.
def feature_importance(over: List[str] = ["PredictorName", "ModelID"]) -> pl.Expr:
    var_imp = weighted_average_polars(
        log_odds(
            pl.col("BinPositives"), pl.col("BinResponseCount") - pl.col("BinPositives")
        ),
        "BinResponseCount",
    ).alias("FeatureImportance")
    if over is not None:
        var_imp = var_imp.over(over)
    return var_imp


def gains_table(
    df: frame,
    value: str,
    index: Optional[str] = None,
    by: Optional[Union[List[str], str]] = None,
) -> pl.DataFrame:
    """Calculates cumulative gains from any data frame.

    The cumulative gains are the cumulative values expressed
    as a percentage vs the size of the population, also expressed
    as a percentage.

    Parameters
    ----------
    df: pl.DataFrame
        The (Polars) dataframe with the raw values
    value: str
        The name of the field with the values (plotted on y-axis)
    index = None
        Optional name of the field for the x-axis. If not passed in
        all records are used and weighted equally.
    by = None
        Grouping field(s), can also be None

    Returns
    -------
    pl.DataFrame
        A (Polars) dataframe with cum_x and cum_y columns and optionally
        the grouping column(s). Values for cum_x and cum_y are relative
        so expressed as values 0-1.

    Examples
    --------
    >>> gains_data = gains_table(df, 'ResponseCount', by=['Channel','Direction])
    """

    sort_expr = pl.col(value) if index is None else pl.col(value) / pl.col(index)
    index_expr = (
        (pl.int_range(1, pl.count() + 1) / pl.count())
        if index is None
        else (pl.cum_sum(index) / pl.sum(index))
    )

    if by is None:
        gains_df = pl.concat(
            [
                pl.DataFrame(data={"cum_x": [0.0], "cum_y": [0.0]}).lazy(),
                df.lazy()
                .sort(sort_expr, descending=True)
                .select(
                    index_expr.cast(pl.Float64).alias("cum_x"),
                    (pl.cum_sum(value) / pl.sum(value)).cast(pl.Float64).alias("cum_y"),
                ),
            ]
        )
    else:
        by_as_list = by if isinstance(by, list) else [by]
        sort_expr = by_as_list + [sort_expr]
        gains_df = (
            df.lazy()
            .sort(sort_expr, descending=True)
            .select(
                by_as_list
                + [
                    index_expr.over(by).cast(pl.Float64).alias("cum_x"),
                    (pl.cum_sum(value) / pl.sum(value))
                    .over(by)
                    .cast(pl.Float64)
                    .alias("cum_y"),
                ]
            )
        )
        # Add entry for the (0,0) point
        gains_df = pl.concat(
            [gains_df.group_by(by).agg(cum_x=pl.lit(0.0), cum_y=pl.lit(0.0)), gains_df]
        ).sort(by_as_list + ["cum_x"])

    return gains_df.collect()


# TODO: perhaps the color / plot utils should move into a separate file
def legend_color_order(fig: go.Figure) -> go.Figure:
    """Orders legend colors alphabetically in order to provide pega color
    consistency among different categories"""

    colorway = [
        "#001F5F",  # dark blue
        "#10A5AC",
        "#F76923",  # orange
        "#661D34",  # wine
        "#86CAC6",  # mint
        "#005154",  # forest
        "#86CAC6",  # mint
        "#5F67B9",  # violet
        "#FFC836",  # yellow
        "#E63690",  # pink
        "#AC1361",  # berry
        "#63666F",  # dark grey
        "#A7A9B4",  # medium grey
        "#D0D1DB",  # light grey
    ]
    colors = []
    for trace in fig.data:
        if trace.legendgroup is not None:
            colors.append(trace.legendgroup)
    colors.sort()

    # https://github.com/pegasystems/pega-datascientist-tools/issues/201
    if len(colors) >= len(colorway):
        return fig

    indexed_colors = {k: v for v, k in enumerate(colors)}
    for trace in fig.data:
        if trace.legendgroup is not None:
            try:
                trace.marker.color = colorway[indexed_colors[trace.legendgroup]]
                trace.line.color = colorway[indexed_colors[trace.legendgroup]]
            except AttributeError:  # pragma: no cover
                pass

    return fig
