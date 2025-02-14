"""
tes/test_dataframe
~~~~~~~~~~~~~~~~~~
"""

import pytest
import pandas as pd
import polars as pl
from pyopenms_viz._dataframe import *

####################
# PandasColumnWrapper


def test_pandas_column_wrapper_getattr():
    # Create a Pandas Series
    series = pd.Series([1, 2, 3, 4, 5])
    
    # Create a PandasColumnWrapper instance
    column_wrapper = PandasColumnWrapper(series)
    
    # Test accessing attributes of the underlying Pandas Series
    assert column_wrapper.sum() == 15
    assert column_wrapper.mean() == 3.0
    assert column_wrapper.max() == 5
    assert column_wrapper.min() == 1

def test_pandas_column_wrapper_getattr_invalid():
    # Create a Pandas Series
    series = pd.Series([1, 2, 3, 4, 5])
    
    # Create a PandasColumnWrapper instance
    column_wrapper = PandasColumnWrapper(series)
    
    # Test accessing invalid attributes of the underlying Pandas Series
    with pytest.raises(AttributeError):
        column_wrapper.invalid_attribute

def test_pandas_column_wrapper_cast():
    # Create a Pandas Series
    series = pd.Series([1, 2, 3, 4, 5])
    
    # Create a PandasColumnWrapper instance
    column_wrapper = PandasColumnWrapper(series)
    
    # Test casting the Series to a different dtype
    casted_series = column_wrapper.astype(float)
    assert casted_series.dtype == float

def test_pandas_column_wrapper_is_duplicated():
    # Create a Pandas Series with duplicate values
    series = pd.Series([1, 2, 3, 4, 5, 1, 2])
    
    # Create a PandasColumnWrapper instance
    column_wrapper = PandasColumnWrapper(series)
    
    # Test checking for duplicate values
    duplicated_series = column_wrapper.duplicated()
    assert duplicated_series.tolist() == [False, False, False, False, False, True, True]

def test_pandas_column_wrapper_to_list():
    # Create a Pandas Series
    series = pd.Series([1, 2, 3, 4, 5])
    
    # Create a PandasColumnWrapper instance
    column_wrapper = PandasColumnWrapper(series)
    
    # Test converting the Series to a list
    series_list = column_wrapper.tolist()
    assert series_list == [1, 2, 3, 4, 5]
    

####################
# PolarsColumnWrapper


def test_polars_column_wrapper_getattr():
    # Create a Polars Series
    series = pl.Series("a", [1, 2, 3, 4, 5])
    
    # Create a PolarsColumnWrapper instance
    column_wrapper = PolarsColumnWrapper(series)
    
    # Test accessing attributes of the underlying Polars Series
    assert column_wrapper.sum() == 15
    assert column_wrapper.mean() == 3.0
    assert column_wrapper.max() == 5
    assert column_wrapper.min() == 1

def test_polars_column_wrapper_cast():
    # Create a Polars Series
    series = pl.Series("a", [1, 2, 3, 4, 5])
    
    # Create a PolarsColumnWrapper instance
    column_wrapper = PolarsColumnWrapper(series)
    
    # Test casting the Series to a different dtype
    casted_series = column_wrapper.astype(pl.Float64)
    assert casted_series.dtype == pl.Float64

def test_polars_column_wrapper_is_duplicated_first():
    series = pl.Series("a", [1, 2, 3, 1, 2])
    column_wrapper = PolarsColumnWrapper(series)
    
    duplicated_series = column_wrapper.duplicated(keep='first')
    expected_result = [False, False, False, True, True]  # First occurrence is kept

    assert duplicated_series.to_list() == expected_result

def test_polars_column_wrapper_is_duplicated_last():
    series = pl.Series("a", [1, 2, 3, 1, 2])
    column_wrapper = PolarsColumnWrapper(series)
    
    duplicated_series = column_wrapper.duplicated(keep='last')
    expected_result = [True, True, False, False, False]  # Last occurrence is kept

    assert duplicated_series.to_list() == expected_result

def test_polars_column_wrapper_is_duplicated_all():
    series = pl.Series("a", [1, 2, 3, 1, 2])
    column_wrapper = PolarsColumnWrapper(series)
    
    duplicated_series = column_wrapper.duplicated(keep=False)
    expected_result = [True, True, False, True, True]  # All duplicates marked

    assert duplicated_series.to_list() == expected_result

def test_polars_column_wrapper_to_list():
    # Create a Polars Series
    series = pl.Series("a", [1, 2, 3, 4, 5])
    
    # Create a PolarsColumnWrapper instance
    column_wrapper = PolarsColumnWrapper(series)
    
    # Test converting the Series to a list
    series_list = column_wrapper.tolist()
    assert series_list == [1, 2, 3, 4, 5]
    

####################
# GroupedDataFrame

def test_grouped_dataframe_init_with_pandas():
    # Create a sample Pandas DataFrame and group it
    df = pd.DataFrame({
        'key': ['A', 'B', 'A', 'B'],
        'value': [1, 2, 3, 4]
    })
    grouped = df.groupby('key')

    gdf = GroupedDataFrame(grouped, is_pandas=True)
    
    assert gdf.is_pandas is True
    assert len(list(gdf)) == 2  # Two groups: A and B

def test_grouped_dataframe_iterate_with_pandas():
    df = pd.DataFrame({
        'key': ['A', 'B', 'A', 'B'],
        'value': [1, 2, 3, 4]
    })
    grouped = df.groupby('key')

    gdf = GroupedDataFrame(grouped, is_pandas=True)
    
    groups = list(gdf)
    
    assert len(groups) == 2  # Should have two groups
    assert groups[0][0] == 'A'  # First group name should be 'A'
    assert isinstance(groups[0][1], UnifiedDataFrame)  # Should return UnifiedDataFrame

def test_grouped_dataframe_sum_with_pandas():
    df = pd.DataFrame({
        'key': ['A', 'B', 'A', 'B'],
        'value': [1, 2, 3, 4]
    })
    grouped = df.groupby('key')

    gdf = GroupedDataFrame(grouped, is_pandas=True)
    
    summed_df = gdf.sum()
    
    assert isinstance(summed_df, UnifiedDataFrame)  # Should return a UnifiedDataFrame
    assert summed_df.data.equals(pd.DataFrame({'key': ['A', 'B'], 'value': [4, 6]}))  # Check summed values

def test_grouped_dataframe_init_with_polars():
    # Create a sample Polars DataFrame and group it
    df = pl.DataFrame({
        'key': ['A', 'B', 'A', 'B'],
        'value': [1, 2, 3, 4]
    })
    grouped = df.group_by('key')

    gdf = GroupedDataFrame(grouped, is_pandas=False)
    
    assert gdf.is_pandas is False
    assert len(list(gdf)) == 2  # Two groups: A and B

def test_grouped_dataframe_iterate_with_polars():
    df = pl.DataFrame({
        'key': ['A', 'B', 'A', 'B'],
        'value': [1, 2, 3, 4]
    })
    grouped = df.group_by('key', maintain_order=True)

    gdf = GroupedDataFrame(grouped, is_pandas=False)
    
    groups = list(gdf)
    
    assert len(groups) == 2  # Should have two groups
    assert groups[0][0] == 'A'  # First group name should be 'A'
    assert isinstance(groups[0][1], UnifiedDataFrame)  # Should return UnifiedDataFrame

def test_grouped_dataframe_sum_with_polars():
    df = pl.DataFrame({
        'key': ['A', 'B', 'A', 'B'],
        'value': [1, 2, 3, 4]
    })
    grouped = df.group_by('key', maintain_order=True)

    gdf = GroupedDataFrame(grouped, is_pandas=False)
    
    summed_df = gdf.sum()
    
    assert isinstance(summed_df, UnifiedDataFrame)  # Should return a UnifiedDataFrame
    expected_result = pl.DataFrame({'key': ['A', 'B'], 'value': [4, 6]})
    print(f"summed_df.data: {summed_df.data}")
    print(f"expected_result: {expected_result}")
    
    assert summed_df.data.equals(expected_result)  # Check summed values
    
    
####################
# UnifiedDataFrame


def test_unified_dataframe():
    pandas_data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    polars_data = PolarsDataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    
    pandas_df = UnifiedDataFrame(pandas_data)
    polars_df = UnifiedDataFrame(polars_data)
    
    assert isinstance(pandas_df, UnifiedDataFrame)
    assert isinstance(polars_df, UnifiedDataFrame)
    
    assert len(pandas_df) == 3
    assert len(polars_df) == 3
    
    assert pandas_df.columns == ["A", "B"]
    assert polars_df.columns == ["A", "B"]
    
    assert pandas_df["A"].tolist() == [1, 2, 3]
    assert polars_df["A"].tolist() == [1, 2, 3]
    
    pandas_df["C"] = [7, 8, 9]
    polars_df["C"] = [7, 8, 9]
    
    assert pandas_df.columns == ["A", "B", "C"]
    assert polars_df.columns == ["A", "B", "C"]
    
    sorted_pandas_df = pandas_df.sort_values("A")
    sorted_polars_df = polars_df.sort_values("A")
    
    assert sorted_pandas_df["A"].tolist() == [1, 2, 3]
    assert sorted_polars_df["A"].tolist() == [1, 2, 3]
    
    assert sorted_pandas_df.reset_index(drop=True).index.tolist() == [0, 1, 2]
    assert sorted_polars_df.reset_index(drop=True).index == [0, 1, 2]
    
    assert sorted_pandas_df.duplicated().tolist() == [False, False, False]
    assert sorted_polars_df.duplicated().to_list() == [False, False, False]

# Test UnifiedDataFrame.plot()
def test_unified_dataframe_plot():
    pandas_data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    polars_data = PolarsDataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    
    pandas_df = UnifiedDataFrame(pandas_data)
    polars_df = UnifiedDataFrame(polars_data)
    
    pandas_plot = pandas_df.plot(x="A", y="B", kind="line")
    polars_plot = polars_df.plot(x="A", y="B", kind="line")
    
    assert isinstance(pandas_plot, object)  # Replace with the actual type of the plot object
    assert isinstance(polars_plot, object)  # Replace with the actual type of the plot object

# Test UnifiedDataFrame.tolist()
def test_unified_dataframe_tolist():
    pandas_data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    polars_data = PolarsDataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    
    pandas_df = UnifiedDataFrame(pandas_data)
    polars_df = UnifiedDataFrame(polars_data)
    
    assert pandas_df.tolist("A") == [1, 2, 3]
    assert polars_df.tolist("A") == [1, 2, 3]

# Test UnifiedDataFrame.to_dict()
def test_unified_dataframe_to_dict():
    pandas_data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    polars_data = PolarsDataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    
    pandas_df = UnifiedDataFrame(pandas_data)
    polars_df = UnifiedDataFrame(polars_data)
    
    assert pandas_df.to_dict(orient="list") == {"A": [1, 2, 3], "B": [4, 5, 6]}
    assert polars_df.to_dict() == {"A": [1, 2, 3], "B": [4, 5, 6]}