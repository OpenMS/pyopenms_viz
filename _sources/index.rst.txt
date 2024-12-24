PyOpenMS-Viz
============

.. toctree::
   :maxdepth: 1
   :glob:
   :hidden:

   Installation
   User Guide
   gallery/index
   API


Welcome to PyOpenMS-Viz Documentation! PyOpenMS-Viz is a visualization package for mass spectrometry data directly from pandas dataframes

Key Features Include:

* **DataFrame Based Plotting** - Plot directly from a pandas dataframe object
* **Interactive and Static Plotting** - Multiple backends supported including matplotlib, bokeh and plotly.
* **Usage Flexibility** - User-friendly web based dashboard for quick visualizations, advanced python package for more complex applications 

Quick Start
***********

Installation
------------


.. code-block:: shell

        pip install pyopenms-viz 


Plotting a Spectrum
-------------------

.. code-block:: python

        import pandas as pd
        ms_data = pd.read_csv("path/to/ms_data.csv")
        pd.set_option("plotting.backend", "ms_bokeh") # try changing backend to "ms_plotly" or "ms_matplotlib"
        ms_data.plot(x="m/z", y="intensity", kind="spectrum") 
        
Plotting a Chromatogram
-----------------------

.. code-block:: python

        import pandas as pd
        ms_data = pd.read_csv("path/to/ms_data.csv")
        pd.set_option("plotting.backend", "ms_bokeh") # try changing backend to "ms_plotly" or "ms_matplotlib"
        ms_data.plot(x="rt", y="intensity", kind="chromatogram")


Support
*******

If you are having issues or would like to propose a new feature, please use the `issues tracker <https://github.com/jcharkow/pyopenms-viz/issues>`_.

License
*******

This project is licensed under the BSD 3-Clause license.

Citation
********
If PyOpenMS-Viz was usefull in your research please cite the following:

