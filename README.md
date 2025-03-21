<div align="center">
    <img src="https://github.com/OpenMS/pyopenms_viz/blob/main/docs/img/pyOpenMSviz_logo_color.png" alt="description" width="300"/>
</div>

# pyOpenMS-viz: The Python Pandas-Based Mass Spectrometry Visualization Library
[![pypipv](https://img.shields.io/pypi/pyversions/pyopenms_viz.svg)](https://img.shields.io/pypi/pyversions/pyopenms_viz)
[![pypiv](https://img.shields.io/pypi/v/pyopenms_viz.svg)](https://img.shields.io/pypi/v/pyopenms_viz
)
[![pypidownload](https://img.shields.io/pypi/dm/pyopenms_viz?color=orange)](https://pypistats.org/packages/pyopenms_viz)
[![readthedocs](https://img.shields.io/readthedocs/pyopenms_viz)](https://pyopenms-viz.readthedocs.io/en/latest/index.html)


pyOpenMS-Viz is a Python library that provides a simple interface for extending the plotting capabilities of Pandas DataFrames for creating static or interactive visualizations of mass spectrometry data. It integrates seamlessly with various plotting library backends (matpotlib, bokeh and plotly) and leverages the power of Pandas for data manipulation and representation.

## Features

- Flexible plotting API that interfaces directly with Pandas DataFrames
- Support for multiple plotting backends: matplotlib (static), bokeh and plotly (interactive)
- Visualization of various mass spectrometry data types, including 1D chromatograms, spectra, and 2D peak maps
- Versatile column selection for easy adaptation to different data formats
- Consistent API across different plotting backends for easy switching between static and interactive plots
- Suitable for use in scripts, Jupyter notebooks, and web applications
- Now supports both pandas and polars DataFrames!
- Interactive plots with zoom, pan, and hover capabilities
- Customizable plot styling and annotations

## Suported Plots
| **Plot Type**   | **Required Dimensions** | **pyopenms_viz Name**                                     | **Matplotlib** | **Bokeh** | **Plotly** |
|-----------------|-------------------------|-----------------------------------------------------------|----------------|-----------|------------|
| Chromatogram    | x, y                    | chromatogram                                              | ✓              | ✓         | ✓          |
| Mobilogram      | x, y                    | mobilogram                                                | ✓              | ✓         | ✓          |
| Spectrum        | x, y                    | spectrum                                                  | ✓              | ✓         | ✓          |
| PeakMap 2D      | x, y, z                 | peakmap                                                   | ✓              | ✓         | ✓          |
| PeakMap 3D      | x, y, z                 | peakmap (plot3d=True)                                     | ✓              |           | ✓          |


## (Recommended) Pip Installation

The recommended way of installing pyopenms_viz is through the Python Package Index (PyPI). We recommend installing pyopenms_viz in its own virtual environment using Anaconda to avoid packaging conflicts.

First create a new environemnt:

```bash
conda create --name=pyopenms_viz python=3.12
conda activate pyopenms_viz 
```
Then in the new environment install pyopenms_viz.

```bash
pip install pyopenms_viz --upgrade
```

## Documentation

Documentation can be found [here](https://pyopenms-viz.readthedocs.io/en/latest/index.html)

## References

- Sing, J., Charkow, J., Walter, A. et al. pyOpenMS-viz: Streamlining Mass Spectrometry Data Visualization with pandas. Journal of Proteome Research, (2025) [https://doi.org/10.1021/acs.jproteome.4c00873](https://pubs.acs.org/doi/10.1021/acs.jproteome.4c00873)
- Pfeuffer, J., Bielow, C., Wein, S. et al. OpenMS 3 enables reproducible analysis of large-scale mass spectrometry data. Nat Methods 21, 365–367 (2024). [https://doi.org/10.1038/s41592-024-02197-7](https://doi.org/10.1038/s41592-024-02197-7)

- Röst HL, Schmitt U, Aebersold R, Malmström L. pyOpenMS: a Python-based interface to the OpenMS mass-spectrometry algorithm library. Proteomics. 2014 Jan;14(1):74-7. [https://doi.org/10.1002/pmic.201300246](https://doi.org/10.1002/pmic.201300246). PMID: [24420968](https://pubmed.ncbi.nlm.nih.gov/24420968/).

## Quick Start

```python
import pandas as pd
import polars as pl
from pyopenms_viz import plot

# Using pandas DataFrame
df = pd.DataFrame({
    'mz': [100, 200, 300],
    'intensity': [1000, 2000, 3000]
})
plot(df, x='mz', y='intensity', kind='spectrum')

# Using polars DataFrame
df_pl = pl.DataFrame({
    'mz': [100, 200, 300],
    'intensity': [1000, 2000, 3000]
})
plot(df_pl, x='mz', y='intensity', kind='spectrum')
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
