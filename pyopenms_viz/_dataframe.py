from __future__ import annotations

from typing import Any, Union, List, Optional
import pandas as pd
import polars as pl
import numpy as np

DataFrameType = Union[pd.DataFrame, pl.DataFrame]

class DataFrameWrapper:
    """
    A wrapper class that provides a unified interface for both pandas and polars DataFrames.
    This allows pyopenms_viz to work with either type without modifying the existing API.
    """
    
    def __init__(self, data: DataFrameType):
        self._data = data
        self._is_polars = isinstance(data, pl.DataFrame)
        
    @property
    def data(self) -> DataFrameType:
        """Get the underlying DataFrame."""
        return self._data
    
    def to_pandas(self) -> pd.DataFrame:
        """Convert the DataFrame to pandas if needed."""
        if self._is_polars:
            return self._data.to_pandas()
        return self._data
    
    def to_polars(self) -> pl.DataFrame:
        """Convert the DataFrame to polars if needed."""
        if not self._is_polars:
            return pl.from_pandas(self._data)
        return self._data
    
    def copy(self) -> 'DataFrameWrapper':
        """Create a copy of the DataFrame."""
        if self._is_polars:
            return DataFrameWrapper(self._data.clone())
        return DataFrameWrapper(self._data.copy())
    
    def get_column(self, col: str) -> np.ndarray:
        """Get a column as numpy array."""
        if self._is_polars:
            return self._data[col].to_numpy()
        return self._data[col].to_numpy()
    
    def set_column(self, col: str, value: Any) -> None:
        """Set a column value."""
        if self._is_polars:
            self._data = self._data.with_columns(pl.Series(col, value))
        else:
            self._data[col] = value
    
    def groupby(self, by: str) -> 'GroupByWrapper':
        """Group the DataFrame by a column."""
        if self._is_polars:
            return GroupByWrapper(self._data.groupby(by), is_polars=True)
        return GroupByWrapper(self._data.groupby(by), is_polars=False)
    
    def fillna(self, value: Any) -> 'DataFrameWrapper':
        """Fill NA/null values."""
        if self._is_polars:
            return DataFrameWrapper(self._data.fill_null(value))
        return DataFrameWrapper(self._data.fillna(value))
    
    def between(self, col: str, left: float, right: float) -> 'DataFrameWrapper':
        """Select rows where column values are between left and right."""
        if self._is_polars:
            mask = (self._data[col] >= left) & (self._data[col] <= right)
            return DataFrameWrapper(self._data.filter(mask))
        return DataFrameWrapper(self._data[self._data[col].between(left, right)])
    
    def max(self, col: str) -> float:
        """Get maximum value of a column."""
        if self._is_polars:
            return self._data[col].max()
        return self._data[col].max()
    
    def min(self, col: str) -> float:
        """Get minimum value of a column."""
        if self._is_polars:
            return self._data[col].min()
        return self._data[col].min()
    
    def iterrows(self):
        """Iterate over DataFrame rows."""
        if self._is_polars:
            for row in self._data.iter_rows(named=True):
                yield row
        else:
            for idx, row in self._data.iterrows():
                yield row
    
    @property
    def columns(self) -> List[str]:
        """Get column names."""
        if self._is_polars:
            return self._data.columns
        return list(self._data.columns)
    
    def __getitem__(self, key: str) -> np.ndarray:
        """Get a column by name."""
        return self.get_column(key)


class GroupByWrapper:
    """Wrapper for grouped DataFrame operations."""
    
    def __init__(self, grouped, is_polars: bool):
        self._grouped = grouped
        self._is_polars = is_polars
    
    def __iter__(self):
        """Iterate over groups."""
        if self._is_polars:
            for name, group in self._grouped.groups():
                yield name, DataFrameWrapper(group)
        else:
            for name, group in self._grouped:
                yield name, DataFrameWrapper(group)
    
    def agg(self, func: dict) -> DataFrameWrapper:
        """Aggregate using the specified functions."""
        if self._is_polars:
            agg_exprs = []
            for col, agg_func in func.items():
                if isinstance(agg_func, str):
                    agg_exprs.append(pl.col(col).agg(agg_func))
                else:
                    agg_exprs.append(pl.col(col).agg(lambda x: agg_func(x.to_numpy())))
            result = self._grouped.agg(agg_exprs)
            return DataFrameWrapper(result)
        else:
            return DataFrameWrapper(self._grouped.agg(func))


def wrap_dataframe(data: DataFrameType) -> DataFrameWrapper:
    """Create a DataFrameWrapper instance from a pandas or polars DataFrame."""
    return DataFrameWrapper(data) 