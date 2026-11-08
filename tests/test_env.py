"""Environment sanity test."""
import sys
from pathlib import Path

def test_imports():
    import numpy
    import pandas
    import sklearn
    import xgboost
    import lightgbm
    import shap
    import fastapi
    import streamlit
    import pydantic
    import yaml

    assert numpy is not None
    assert pandas is not None
    assert sklearn is not None
    assert xgboost is not None
    assert lightgbm is not None
    assert shap is not None
    assert fastapi is not None
    assert streamlit is not None
    assert pydantic is not None
    assert yaml is not None


def test_constants():
    from config import constants
    assert constants.RANDOM_SEED == 42
    assert len(constants.POST_ORIGINATION_COLS) > 0
