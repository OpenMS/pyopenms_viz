from pandas.core.dtypes.generic import ABCDataFrame as PandasDataFrame
from pandas.core.groupby.generic import DataFrameGroupBy as PandasGroupBy
from polars.dataframe.frame import DataFrame as PolarsDataFrame
from polars.series import Series as PolarsSeries
from polars.dataframe.group_by import GroupBy as PolarsGroupBy

import polars as pl


class PandasColumnWrapper:
    """Wrapper for Pandas Series to add custom methods."""
    def __init__(self, series):
        self.series = series
        
    def __getattr__(self, name):
        """Delegate attribute access to the underlying Pandas Series."""
        return getattr(self.series, name)


class PolarsColumnWrapper:
    """Wrapper for Polars Series to add custom methods."""
    def __init__(self, series):
        self.series = series
        
    def __getattr__(self, name):
        """Delegate attribute access to the underlying Polars Series."""
        return getattr(self.series, name)
    
    def duplicated(self):
        """Return a boolean Series indicating duplicate values."""
        return self.series.is_duplicated()

    def tolist(self):
        """Return the Series as a list."""
        return self.series.to_list()
   
 
class UnifiedDataFrame:
    """
    Wrapper class for Pandas and Polars DataFrames to provide a unified interface.
    """
    def __init__(self, data):
        if isinstance(data, (PandasDataFrame, PolarsDataFrame)):
            self.data = data
        else:
            raise TypeError("Unsupported data type. Must be either pandas DataFrame or Polars DataFrame.")
        
    def __getitem__(self, key):
        """Allow access to columns using bracket notation."""
        if isinstance(self.data, PandasDataFrame):
            return PandasColumnWrapper(self.data[key])  
        elif isinstance(self.data, PolarsDataFrame):
            return PolarsColumnWrapper(self.data[key])  
        else:
            raise KeyError(f"Column '{key}' not found in DataFrame.")
    
    def __setitem__(self, key, value):
        """Allow assignment to columns using bracket notation."""
        if isinstance(self.data, PandasDataFrame):
            self.data[key] = value  
        elif isinstance(self.data, PolarsDataFrame):
            self.data = self.data.with_columns(
                PolarsSeries(key, value)  
            )
            
    def __len__(self):
        """Return the number of rows in the DataFrame."""
        if isinstance(self.data, PandasDataFrame):
            return len(self.data)
        elif isinstance(self.data, PolarsDataFrame):
            return self.data.height
    
    @property
    def columns(self):
        """Return a list of column names."""
        if isinstance(self.data, PandasDataFrame):
            return self.data.columns.tolist()
        elif isinstance(self.data, PolarsDataFrame):
            return self.data.columns

    def copy(self):
        """Return a copy of the DataFrame."""
        if isinstance(self.data, PandasDataFrame):
            return UnifiedDataFrame(self.data.copy())
        elif isinstance(self.data, PolarsDataFrame):
            return UnifiedDataFrame(self.data.clone())

    def sort_values(self, by, ascending=True):
        """Sort the DataFrame by the specified column(s)."""
        if isinstance(self.data, PandasDataFrame):
            return UnifiedDataFrame(self.data.sort_values(by=by, ascending=ascending).reset_index(drop=True))
        elif isinstance(self.data, PolarsDataFrame):
            return UnifiedDataFrame(
                self.data.sort(by=by, descending=not ascending).with_row_count().rename({"row_nr": "index"})
            )
            
    def reset_index(self, drop=False):
        """Reset the index of the DataFrame."""
        if isinstance(self.data, PandasDataFrame):
            return UnifiedDataFrame(self.data.reset_index(drop=drop))
        elif isinstance(self.data, PolarsDataFrame):
            # For Polars we can just return the same DataFrame since it doesn't have an index like Pandas.
            return UnifiedDataFrame(self.data)  

    def iterrows(self):
        """Return an iterator for rows of the DataFrame."""
        if isinstance(self.data, PandasDataFrame):
            return self.data.iterrows()
        elif isinstance(self.data, PolarsDataFrame):
            return enumerate(self.data.iter_rows(named=True))
        
    def groupby(self, by):
        """Group by specified columns."""
        if isinstance(self.data, PolarsDataFrame):
            return UnifiedDataFrame(self.data.groupby(by))  
        elif isinstance(self.data, PolarsDataFrame):
            return UnifiedDataFrame(self.data.groupby(by))  

    def sum(self):
        """Sum the grouped data."""
        if isinstance(self.data, PandasGroupBy):  
            return UnifiedDataFrame(self.data.sum().reset_index())  
        elif isinstance(self.data, PolarsGroupBy):  
            return UnifiedDataFrame(self.data.agg(pl.sum(pl.col("*"))))  
        
    def tolist(self, column_name):
        """Return a list of values from a specified column."""
        if isinstance(self.data, PandasDataFrame):
            return self.data[column_name].tolist()
        elif isinstance(self.data, PolarsDataFrame):
            return self.data[column_name].to_list()