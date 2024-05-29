from pathlib import Path
from typing import (
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import polars as pl

from .. import pega_io
from ..utils import cdh_utils
from .table_definitions import (
    ADMModelSnapshot,
    ADMPredictorBinningSnapshot,
    TableConfig,
)

frame = TypeVar("frame", pl.DataFrame, pl.LazyFrame)


def import_datamart(
    context_keys: list[str],
    path: Union[str, Path] = ".",
    import_strategy: Literal["eager", "lazy"] = "eager",
    *,
    model_filename: str = "model_data",
    predictor_filename: str = "predictor_data",
    model_df: Optional[frame] = None,
    predictor_df: Optional[frame] = None,
    query: Optional[Union[pl.Expr, List[pl.Expr], str, Dict[str, list]]] = None,
    subset: bool = True,
    drop_cols: Optional[list] = None,
    include_cols: Optional[list] = None,
    extract_keys: bool = False,
    **reading_opts,
) -> Tuple[
    Union[pl.LazyFrame, None], Union[pl.LazyFrame, None], Union[pl.LazyFrame, None]
]:
    # First we read in both files
    model_data = model_df or pega_io.read_ds_export(
        filename=model_filename, path=path, **reading_opts
    )
    predictor_data = predictor_df or pega_io.read_ds_export(
        filename=predictor_filename, path=path, **reading_opts
    )

    # We then check whether any of them are DataFrames - this means they have been
    # fully read and are in-memory, so we should update import_strategy accordingly
    if isinstance(model_data, pl.DataFrame) or isinstance(predictor_data, pl.DataFrame):
        import_strategy = "eager"

    model_data = process(model_data, "models", subset, include_cols, drop_cols).lazy()
    if model_data:
        model_data = postprocess_models(
            model_data,
            extract_keys=extract_keys,
            import_strategy=import_strategy,
            context_keys=context_keys,
        )
    predictor_data = process(
        predictor_data, "predictors", subset, include_cols, drop_cols
    ).lazy()

    combined_data = join_datamart_tables(model_data, predictor_data)
    return model_data, predictor_data, combined_data


def postprocess_models(
    df: frame,
    extract_keys: bool,
    import_strategy: Literal["eager", "lazy"],
    context_keys: List[str],
) -> frame:
    if extract_keys:
        df = df.with_columns(
            cdh_utils._extract_keys(df, import_strategy=import_strategy)
        )
    return df.with_columns(
        pl.lit("NA").alias(key) for key in context_keys if key not in df.columns
    ).with_columns(
        (pl.col("Positives") / pl.col("ResponseCount"))
        .fill_nan(pl.lit(0))
        .alias("SuccessRate"),
        last_timestamp("Positives"),
        last_timestamp("ResponseCount"),
    )


def process(
    df: frame,
    table: Literal["models", "predictors"],
    subset: bool = True,
    include_cols: Optional[Iterable[str]] = None,
    drop_cols: Optional[Iterable[str]] = None,
) -> frame:
    df = cdh_utils._polars_capitalize(df)
    table_definition = get_table_definition(table)

    type_map = get_schema(
        df,
        table_definition=table_definition,
        include_cols=include_cols or {},
        drop_cols=drop_cols or {},
        subset=subset,
    )
    df = df.select(pl.col(name).cast(_type) for name, _type in type_map.items())
    if "SnapshotTime" not in df.columns:
        df = df.with_columns(SnapshotTime=None)
    return df


def get_table_definition(table: Literal["models", "predictors"]):
    mapping = {"models": ADMModelSnapshot}
    if table not in mapping:
        raise ValueError(f"Unknown table: {table}")
    return mapping[table]


def get_schema(
    df: frame,
    table_definition: Dict[str, TableConfig],
    include_cols: Iterable[str],
    drop_cols: Iterable[str],
    subset: bool,
) -> Dict[str, Type[pl.DataType]]:
    type_map: Dict[str, Type[pl.DataType]] = dict()
    checked_columns: Set[str] = set()

    for column, config in table_definition.items():
        df_col = None
        checked_columns = checked_columns.union({column, config["label"]})
        if column in df.columns:
            df_col = column

        elif config["label"] in df.columns:
            df_col = config["label"]

        if (
            df_col is not None  # If we've found a matching column
            and (
                subset is False  # We don't want to take a subset...
                or (
                    df_col not in drop_cols  # The user does not want to drop it
                    and (config["default"] or df_col in include_cols)
                )  # And it's either default or part of include cols
            )
        ):
            # Then we make it part of the type map, which we use to filter down
            type_map[df_col] = config["type"]

    unknown_columns = [col for col in df.columns if col not in checked_columns]
    if unknown_columns:
        raise ValueError("Unknown columns found: ", unknown_columns)

    return type_map


def last_timestamp(col: Literal["ResponseCount", "Positives"]) -> pl.Expr:
    """Add a column to indicate the last timestamp a column has changed.

    Parameters
    ----------
    col : Literal['ResponseCount', 'Positives']
        The column to calculate the diff for
    """

    return (
        pl.when(pl.col(col).min() == pl.col(col).max())
        .then(pl.max("SnapshotTime"))
        .otherwise(pl.col("SnapshotTime").filter(pl.col(col).diff() != 0).max())
        .over("ModelID")
        .alias(f"Last_{col}")
    )


def join_datamart_tables(model_data, predictor_data) -> pl.LazyFrame:
    ...
