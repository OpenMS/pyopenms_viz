# Python Pandas-Based OpenMS Visualization Library
[![pypipv](https://img.shields.io/pypi/pyversions/pyopenms_viz.svg)](https://img.shields.io/pypi/pyversions/pyopenms_viz)
[![pypiv](https://img.shields.io/pypi/v/pyopenms_viz.svg)](https://img.shields.io/pypi/v/pyopenms_viz
)
[![pypidownload](https://img.shields.io/pypi/dm/pyopenms_viz?color=orange)](https://pypistats.org/packages/pyopenms_viz)

pyopenms_viz is a Python library that provides a simple interface for extending the plotting capabilities of Pandas DataFrames for creating static or interactive visualizations of mass spectrometry data. It integrates seamlessly with various plotting library backends (matpotlib, bokeh and plotly) and leverages the power of Pandas for data manipulation and representation.

## Features

- Flexible plotting API that interfaces directly with Pandas DataFrames
- Support for multiple plotting backends: matplotlib (static), bokeh and plotly (interactive)
- Visualization of various mass spectrometry data types, including 1D chromatograms, spectra, and 2D peak maps
- Versatile column selection for easy adaptation to different data formats
- Consistent API across different plotting backends for easy switching between static and interactive plots
- Suitable for use in scripts, Jupyter notebooks, and web applications

## (Recommended) Pip Installation

The recommended way of installing pyopenms_viz is through the Python Package Index (PyPI). We recommend installing pyopenms_viz in its own virtual environment using Anaconda to avoid packaging conflicts.

First create a new environemnt:

```bash
conda create --name=pyopenms_viz python=3.10
conda activate pyopenms_viz 
```
Then in the new environment install pyopenms_viz.

```bash
pip install pyopenms_viz --upgrade
```

## Documentation

Documentation (*work in progress*).

## References

- Pfeuffer, J., Bielow, C., Wein, S. et al. OpenMS 3 enables reproducible analysis of large-scale mass spectrometry data. Nat Methods 21, 365–367 (2024). [https://doi.org/10.1038/s41592-024-02197-7](https://doi.org/10.1038/s41592-024-02197-7)

- Röst HL, Schmitt U, Aebersold R, Malmström L. pyOpenMS: a Python-based interface to the OpenMS mass-spectrometry algorithm library. Proteomics. 2014 Jan;14(1):74-7. [https://doi.org/10.1002/pmic.201300246](https://doi.org/10.1002/pmic.201300246). PMID: [24420968](https://pubmed.ncbi.nlm.nih.gov/24420968/).
