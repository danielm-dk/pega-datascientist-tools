from typing import Dict, Type, TypedDict

import polars as pl
from pydantic import BaseModel, Field


class TableConfig(TypedDict):
    label: str
    default: bool
    type: Type[pl.DataType]


ADMModelSnapshot: Dict[str, TableConfig] = {
    "pxApplication": {"label": "Application", "default": False, "type": pl.Categorical},
    "pyAppliesToClass": {
        "label": "AppliesToClass",
        "default": False,
        "type": pl.Categorical,
    },
    "pyModelID": {"label": "ModelID", "default": True, "type": pl.Utf8},
    "pyConfigurationName": {
        "label": "Configuration",
        "default": True,
        "type": pl.Categorical,
    },
    "pySnapshotTime": {"label": "SnapshotTime", "default": True, "type": pl.Datetime},
    "pyIssue": {"label": "Issue", "default": True, "type": pl.Categorical},
    "pyGroup": {"label": "Group", "default": True, "type": pl.Categorical},
    "pyName": {"label": "Name", "default": True, "type": pl.Utf8},
    "pyChannel": {"label": "Channel", "default": True, "type": pl.Categorical},
    "pyDirection": {"label": "Direction", "default": True, "type": pl.Categorical},
    "pyTreatment": {"label": "Treatment", "default": True, "type": pl.Utf8},
    "pyPerformance": {"label": "Performance", "default": True, "type": pl.Float64},
    "pySuccessRate": {"label": "SuccessRate", "default": True, "type": pl.Float64},
    "pyResponseCount": {"label": "ResponseCount", "default": True, "type": pl.Float32},
    "pxObjClass": {"label": "ObjClass", "default": False, "type": pl.Categorical},
    "pzInsKey": {"label": "InsKey", "default": False, "type": pl.Utf8},
    "pxInsName": {"label": "InsName", "default": False, "type": pl.Utf8},
    "pxSaveDateTime": {"label": "SaveDateTime", "default": False, "type": pl.Datetime},
    "pxCommitDateTime": {
        "label": "CommitDateTime",
        "default": False,
        "type": pl.Datetime,
    },
    "pyExtension": {"label": "Extension", "default": False, "type": pl.Utf8},
    "pyActivePredictors": {
        "label": "ActivePredictors",
        "default": True,
        "type": pl.UInt16,
    },
    "pyTotalPredictors": {
        "label": "TotalPredictors",
        "default": True,
        "type": pl.UInt16,
    },
    "pyNegatives": {"label": "Negatives", "default": True, "type": pl.Float32},
    "pyPositives": {"label": "Positives", "default": True, "type": pl.Float32},
    "pyRelativeNegatives": {
        "label": "RelativeNegatives",
        "default": False,
        "type": pl.Float32,
    },
    "pyRelativePositives": {
        "label": "RelativePositives",
        "default": False,
        "type": pl.Float32,
    },
    "pyRelativeResponseCount": {
        "label": "RelativeResponseCount",
        "default": False,
        "type": pl.Float32,
    },
    "pyMemory": {"label": "Memory", "default": False, "type": pl.Utf8},
    "pyPerformanceThreshold": {
        "label": "PerformanceThreshold",
        "default": False,
        "type": pl.Float32,
    },
    "pyCorrelationThreshold": {
        "label": "CorrelationThreshold",
        "default": False,
        "type": pl.Float32,
    },
    "pyPerformanceError": {
        "label": "PerformanceError",
        "default": False,
        "type": pl.Float32,
    },
    "pyModelData": {"label": "Modeldata", "default": False, "type": pl.Utf8},
    "pyModelVersion": {"label": "ModelVersion", "default": False, "type": pl.Utf8},
    "pyFactoryUpdatetime": {
        "label": "FactoryUpdateTime",
        "default": False,
        "type": pl.Datetime,
    },
}

# class ADMModelSnapshot(TypedDict):
#     pxApplication: pl.Categorical
#     pyAppliesToClass: pl.Categorical
#     pyModelID = pl.Utf8
#     pyConfigurationName = pl.Categorical
#     pySnapshotTime = pl.Datetime
#     pyIssue = pl.Categorical
#     pyGroup = pl.Categorical
#     pyName = pl.Utf8
#     pyChannel = pl.Categorical
#     pyDirection = pl.Categorical
#     pyTreatment = pl.Utf8
#     pyPerformance = pl.Float64
#     pySuccessRate = pl.Float64
#     pyResponseCount = pl.Float32
#     pxObjClass = pl.Categorical
#     pzInsKey = pl.Utf8
#     pxInsName = pl.Utf8
#     pxSaveDateTime = pl.Datetime
#     pxCommitDateTime = pl.Datetime
#     pyExtension = pl.Utf8
#     pyActivePredictors = pl.UInt16
#     pyTotalPredictors = pl.UInt16
#     pyNegatives = pl.Float32
#     pyPositives = pl.Float32
#     pyRelativeNegatives = pl.Float32
#     pyRelativePositives = pl.Float32
#     pyRelativeResponseCount = pl.Float32
#     pyMemory = pl.Utf8
#     pyPerformanceThreshold = pl.Float32
#     pyCorrelationThreshold = pl.Float32
#     pyPerformanceError = pl.Float32
#     pyModelData = pl.Utf8
#     pyModelVersion = pl.Utf8
#     pyFactoryUpdatetime = pl.Datetime


class ADMPredictorBinningSnapshot:
    pxCommitDateTime = pl.Datetime
    pxSaveDateTime = pl.Datetime
    pyModelID = pl.Utf8
    pxObjClass = pl.Categorical
    pzInsKey = pl.Utf8
    pxInsName = pl.Utf8
    pyPredictorName = pl.Categorical
    pyContents = pl.Utf8
    pyPerformance = pl.Float64
    pyPositives = pl.Float32
    pyNegatives = pl.Float32
    pyType = pl.Categorical
    pyTotalBins = pl.UInt16
    pyResponseCount = pl.Float32
    pyRelativePositives = pl.Float32
    pyRelativeNegatives = pl.Float32
    pyRelativeResponseCount = pl.Float32
    pyBinNegatives = pl.Float32
    pyBinPositives = pl.Float32
    pyBinType = pl.Categorical
    pyBinNegativesPercentage = pl.Float32
    pyBinPositivesPercentage = pl.Float32
    pyBinSymbol = pl.Utf8
    pyBinLowerBound = pl.Float32
    pyBinUpperBound = pl.Float32
    pyRelativeBinPositives = pl.Float32
    pyRelativeBinNegatives = pl.Float32
    pyBinResponseCount = pl.Float32
    pyRelativeBinResponseCount = pl.Float32
    pyBinResponseCountPercentage = pl.Float32
    pySnapshotTime = pl.Datetime
    pyBinIndex = pl.UInt16
    pyLift = pl.Float64
    pyZRatio = pl.Float64
    pyEntryType = pl.Categorical
    pyExtension = pl.Utf8
    pyGroupIndex = pl.UInt32
    pyCorrelationPredictor = pl.Float32
