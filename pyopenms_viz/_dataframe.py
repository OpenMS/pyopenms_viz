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
    
    def astype(self, dtype):
        """Cast the Series to the specified dtype."""
        return self.series.cast(dtype)
    
    def duplicated(self, keep='first'):
        """Return a boolean Series indicating duplicate values."""
        duplicated_mask = self.series.is_duplicated()
        if keep == 'first':
            first_occurrences = self.series.is_first_distinct()
            return duplicated_mask & ~first_occurrences

        elif keep == 'last':
            last_occurrences = self.series.is_last_distinct()
            return duplicated_mask & ~last_occurrences

        elif keep is False:
            return duplicated_mask.cast(pl.Boolean)

        else:
            raise ValueError("keep must be 'first', 'last', or False")

    def tolist(self):
        """Return the Series as a list."""
        return self.series.to_list()
   
class GroupedDataFrame:
    """Class to handle grouped DataFrames for both Pandas and Polars."""
    def __init__(self, grouped_data, is_pandas=True):
        self.grouped_data = grouped_data
        self.is_pandas = is_pandas
        
    def __iter__(self):
        """Allow iteration over groups."""
        if self.is_pandas:
            for group_name, group_df in self.grouped_data:
                yield group_name, UnifiedDataFrame(group_df)
        else:
            for group_name, group_df in self.grouped_data:
                yield group_name[0], UnifiedDataFrame(group_df)

    def sum(self):
        """Sum the grouped data."""
        if self.is_pandas:
            summed_data = self.grouped_data.sum().reset_index()  
            return UnifiedDataFrame(summed_data)
        else:
            summed_data = self.grouped_data.agg(pl.all().sum())
            return UnifiedDataFrame(summed_data)

@pl.api.register_dataframe_namespace("mass") 
class UnifiedDataFrame:
    """
    Wrapper class for Pandas and Polars DataFrames to provide a unified interface.
    """
    def __init__(self, data):
        if isinstance(data, (PandasDataFrame, PolarsDataFrame)):
            self.data = data
        else:
            raise TypeError(f"Unsupported data type {type(data)}. Must be either pandas DataFrame or Polars DataFrame.")
        
    def __repr__(self):
        """Return a string representation of the DataFrame."""
        if isinstance(self.data, PandasDataFrame):
            return str(self.data)  
        elif isinstance(self.data, PolarsDataFrame):
            return self.data.__str__()
        
    def __getitem__(self, key):
        """Allow access to columns using bracket notation."""
        if isinstance(self.data, PandasDataFrame):
            if isinstance(key, list):  
                return UnifiedDataFrame(self.data[key])  
            else:  
                return PandasColumnWrapper(self.data[key])
        elif isinstance(self.data, PolarsDataFrame):
            if isinstance(key, list):  
                return UnifiedDataFrame(self.data.select(key)) 
            else:  
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
    def index(self):
        """Return the index of the DataFrame."""
        if isinstance(self.data, PandasDataFrame):
            return self.data.index  
        elif isinstance(self.data, PolarsDataFrame):
            return list(range(self.data.height)) 
    
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

    def sort_values(self, by, ascending=True, inplace=False):
        """Sort the DataFrame by the specified column(s)."""
        if isinstance(self.data, PandasDataFrame):  
            if inplace:
                self.data.sort_values(by=by, ascending=ascending, inplace=True)
            else:
                sorted_data = self.data.sort_values(by=by, ascending=ascending)
                return UnifiedDataFrame(sorted_data)
        
        elif isinstance(self.data, PolarsDataFrame):  
            sorted_data = self.data.sort(by=by, descending=not ascending)
            if inplace:
                self.data = sorted_data
            else:
                return UnifiedDataFrame(sorted_data)
            
    def reset_index(self, drop=False):
        """Reset the index of the DataFrame."""
        if isinstance(self.data, PandasDataFrame):
            return UnifiedDataFrame(self.data.reset_index(drop=drop))
        elif isinstance(self.data, PolarsDataFrame):
            # For Polars we can just return the same DataFrame since it doesn't have an index like Pandas.
            return UnifiedDataFrame(self.data)  
        
    def duplicated(self):
        """Return a boolean Series indicating duplicate rows."""
        if isinstance(self.data, PandasDataFrame):
            return self.data.duplicated()
        elif isinstance(self.data, PolarsDataFrame):
            return self.data.is_duplicated()

    def iterrows(self):
        """Return an iterator for rows of the DataFrame."""
        if isinstance(self.data, PandasDataFrame):
            return self.data.iterrows()
        elif isinstance(self.data, PolarsDataFrame):
            return enumerate(self.data.iter_rows(named=True))
        
    def groupby(self, by, sort=True):
        """Group by specified columns."""
        if isinstance(self.data, PandasDataFrame):
            grouped = self.data.groupby(by, sort=sort)
            return GroupedDataFrame(grouped)
        elif isinstance(self.data, PolarsDataFrame):
            grouped = self.data.group_by(by)
            return GroupedDataFrame(grouped, is_pandas=False)

    def plot(self, x: str, y: str, kind: str = "line", **kwargs):
        from ._core import PlotAccessor
        return PlotAccessor(self.data)(x, y, kind, **kwargs)
    
    def tolist(self, column_name):
        """Return a list of values from a specified column."""
        if isinstance(self.data, PandasDataFrame):
            return self.data[column_name].tolist()
        elif isinstance(self.data, PolarsDataFrame):
            return self.data[column_name].to_list()
        
    def to_dict(self, orient='list'):
        """Return the DataFrame as a dictionary."""
        if isinstance(self.data, PandasDataFrame):
            return self.data.to_dict(orient=orient)
        elif isinstance(self.data, PolarsDataFrame):
            return self.data.to_dict(as_series=False)