from enum import Enum
from typing import List, Union
import pandas as pd

""" Outlines Column Types. Other columns are permitted in the class however they will be ignored """

class DataType(Enum):
    NUMERIC=0
    STRING=1

class ColumnType(Enum):
    REQUIRED=0 # Plots cannot be generated without these columns
    OPTIONAL=1 # Used to add supplementary information to the plot, but not required for plotting the object

class ColumnSchema:
    def __init__(self, 
                 name: Union[str, List[str]],
                 dataType: DataType = None, 
                 description: str = None, 
                 columnType: ColumnType = ColumnType.OPTIONAL):

        """Define a column schema for a table, although 

        Args:
            name (str): Name of column
            dataType (DataType, Optional): Datatype of column. Defaults to None.
            description (str, Optional): Description of information that column is storing. Defaults to None.
            columnType (ColumnType, optional): Whether column is REQUIRED or OPTIONAL. Defaults to OPTIONAL
        """

        self.name = name
        self.dataType = dataType
        self.description = description
        self.columnType = columnType

    def __str__(self):
        return f"ColumnSchema: {self.name} ({self.dataType})"
    
class DataFrameSchema:
    def __init__(self, 
                 name: str,
                 columns: List[ColumnSchema],
                 description: str = None):

        """Define a table schema

        Args:
            columns (List[ColumnSchema]): List of ColumnSchema objects
            description (str, Optional): Description of table. Defaults to None.
        """
        self.name = name
        self.columns = columns
        self.description = description

    def __str__(self):
        return f"GenericTableSchema: {self.name}. Required Columns: {[i.name for i in self.getRequiredColumns()]}" 
    
    def getOptionalColumns(self):
        return [col for col in self.columns if col.columnType == ColumnType.OPTIONAL]
    
    def getRequiredColumns(self):
        return [col for col in self.columns if col.columnType == ColumnType.REQUIRED]
    
    def validateDataFrame(self, df):
        for col in self.getRequiredColumns():
            # if column name is a list, then exactly one column must be present
            if isinstance(col.name, List):
                intersect = set(df.columns).intersection(set(col.name)) 
                if len(intersect) == 0:
                    raise ValueError(f"Column {col.name} is required for {self.name} schema")
                elif len(intersect) > 1:
                    raise ValueError(f"Only one of {col.name} columns is allowed for {self.name} schema")
                nameFound = False
            else: # assume column name is a string
                if col.name not in df.columns:
                    raise ValueError(f"Column {col.name} is required for {self.name} schema")

            if col.dataType == DataType.NUMERIC:
                assert pd.str.isnumeric(df[col.name])

        for col in self.getOptionalColumns():
            if col.dataType == DataType.NUMERIC:
                assert pd.str.isnumeric(df[col.name])
        return True
    

############################################################################################################
#### DEFINED CHROMATOGRAM SCHEMAS ####
############################################################################################################

REQUIRED_CHROMATOGRAM_COLUMNS = [
        ColumnSchema("time", 
                     dataType=DataType.NUMERIC, 
                     description="Numeric column representing the retention time (in seconds) of the chromatographic peaks.",
                     columnType=ColumnType.REQUIRED), 

        ColumnSchema("intensity",
                     dataType=DataType.NUMERIC, 
                     description="Numeric column representing the intensity (abundance) of the signal at each time point.", 
                     columnType=ColumnType.REQUIRED)]

MULTIPLE_CHROMATOGRAMS_COLUMNS = REQUIRED_CHROMATOGRAM_COLUMNS + [ ColumnSchema("label", 
                                                                                dataType=DataType.STRING, 
                                                                                description="Chromatogram label, necessary if multiple chromatograms are in the same dataframe. This column is used to label the chromatograms", columnType=ColumnType.REQUIRED)]

CHROMATOGRAM = DataFrameSchema(name='chromatogram', columns=REQUIRED_CHROMATOGRAM_COLUMNS, description="Base chromatogram schema")
MULTIPLE_CHROMATOGRAMS = DataFrameSchema(name='multiple chromatograms', columns=REQUIRED_CHROMATOGRAM_COLUMNS + REQUIRED_CHROMATOGRAM_COLUMNS, description="Multiple chromatograms in a single DataFrame schema")

############################################################################################################
#### DEFINED SPECTRUM SCHEMAS ####
############################################################################################################
REQUIRED_SPECTRUM_COLUMNS = [
    ColumnSchema("mz",
                 dataType=DataType.NUMERIC,
                 description="Numeric column representing the mass-to-charge ratio (m/z) values of the peaks in the mass spectrum.",
                 columnType=ColumnType.REQUIRED),
    
    ColumnSchema("intensity",
                 dataType=DataType.NUMERIC,
                 description="Numeric column representing the intensity (abundance) of the peaks in the mass spectrum.",
                 columnType=ColumnType.REQUIRED),
]

MULTIPLE_SPECTRA_COLUMNS = REQUIRED_SPECTRUM_COLUMNS + [ ColumnSchema(name=["label", "native_id"],
                                                                      dataType=DataType.STRING,
                                                                      description="Spectrum label, necessary if multiple spectra are in the same dataframe. This column is used to label the spectra", columnType=ColumnType.REQUIRED)]

SPECTRUM = DataFrameSchema(name='spectrum', columns=REQUIRED_SPECTRUM_COLUMNS, description="Base spectrum schema, only a single spectrum is present")
MULTIPLE_SPECTRA = DataFrameSchema(name='multiple spectra', columns=REQUIRED_SPECTRUM_COLUMNS + REQUIRED_SPECTRUM_COLUMNS, description="Multiple spectra in a single DataFrame")

############################################################################################################
#### DEFINED CHROMATOGRAM FEATURE SCHEMAS ####
############################################################################################################
REQUIRED_CHROMATOGRAM_FEATURE_COLUMNS = [ ColumnSchema(name=['RT', 'rt_apex'],
                                                       dataType=DataType.NUMERIC,
                                                       description="Numeric column representing the retention time of the peak apex.")]

# Note: only include columns that could be used in plotting (e.g. exclude area because this will not be plotted)
CHROMATOGRAM_FEATURE = DataFrameSchema(name='chromatogram feature', 
                                       columns=[
                                           ColumnSchema(name=["rt_apex", 'RT'],
                                           dataType=DataType.NUMERIC,
                                           description="Numeric column representing the retention time of the peak apex."),

                                           ColumnSchema(name=["feature_id", 'id', 'label'],
                                                        dataType=DataType.STRING,
                                                        description="Represent the Id of the feature",
                                                        columnType=ColumnType.OPTIONAL),

                                           ColumnSchema(name=["left_width", 'left_boundary'],
                                                        dataType=DataType.NUMERIC,
                                                        description="Numeric column representing the left boundary of the peak.",
                                                        columnType=ColumnType.OPTIONAL),
                                        
                                           ColumnSchema(name=["right_width", 'right_boundary'],
                                                        dataType=DataType.NUMERIC,
                                                        description="Numeric column representing the right boundary of the peak.",
                                                        columnType=ColumnType.OPTIONAL),
                                            
                                            ColumnSchema(name=["q_value"],
                                                         dataType=DataType.NUMERIC,
                                                         description="Numeric column representing the q-value of the peak.",
                                                         columnType=ColumnType.OPTIONAL),

                                            ColumnSchema(name=["rank"],
                                                         dataType=DataType.NUMERIC,
                                                         description="Numeric column representing the rank of the peak. (1 is best rank)",
                                                         columnType=ColumnType.OPTIONAL)
                                        ],
                                        description="Chromatogram feature schema")

############################################################################################################
#### DEFINED SPECTRUM FEATURE SCHEMAS ####
############################################################################################################
##TODO, someone more familiar with the data should fill this in
OPTIONAL_METADATA_SPECTRUM_DATAFRAME_COLUMNS = {
    "sequence": "String column representing the peptide sequence.",
    "modified_sequence": "String column representing the modified peptide sequence.",
    "precursor_mz": "Numeric column representing the mass-to-charge ratio (m/z) of the precursor ion.",
    "precursor_charge": "Integer column representing the charge state of the precursor ion.",
    "product_mz": "Numeric column representing the mass-to-charge ratio (m/z) of the product ion.",
    "product_charge": "Integer column representing the charge state of the product ion.",
    "annotation": "String column representing the annotation of the spectrum, such as the fragment ion series."
}

