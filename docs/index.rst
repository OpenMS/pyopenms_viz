PyOpenMS-Viz
============

.. |pypiv| image:: https://img.shields.io/pypi/v/massdash.svg
   :target: https://pypi.python.org/pypi/massdash

.. |pypidownload| image:: https://img.shields.io/pypi/dm/massdash?color=orange
   :target: https://pypistats.org/packages/massdash

.. |Python| image:: https://img.shields.io/pypi/pyversions/massdash.svg 
   :target: https://www.python.org/downloads/

.. |dockerv| image:: https://img.shields.io/docker/v/singjust/massdash?label=docker&color=green
   :target: https://hub.docker.com/r/singjust/massdash

.. |dockerpull| image:: https://img.shields.io/docker/pulls/singjust/massdash?color=green
   :target: https://hub.docker.com/r/singjust/massdash

.. |Licence| image:: https://img.shields.io/badge/License-BSD_3--Clause-orange.svg
   :target: https://raw.githubusercontent.com/RoestLab/massdash/main/LICENSE

|pypiv| |pypidownload| |Python| |dockerv| |dockerpull| |Licence|


Welcome to PyOpenMS-Viz Documentation! PyOpenMS-Viz is a visualization package for mass spectrometry data directly from pandas dataframes
Welcome to MassDash Documentation! MassDash is a visualization and data exploration platform for Data-Independent Acquisition mass spectrometry data. 

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


User Guide
**********

.. toctree::
   :maxdepth: 1
   :glob:

   Installation
   User Guide
   Testing
   PlottingGallery
   API

Contribute
**********

* `Issues Tracker <https://github.com/jcharkow/pyopenms-viz/issues>`_
* `Source Code <https://github.com/jcharkow/pyopenms-viz/>`_

Support
*******

If you are having issues or would like to propose a new feature, please use the `issues tracker <https://github.com/jcharkow/pyopenms-viz/issues>`_.

License
*******

This project is licensed under the BSD 3-Clause license.

Citation
********
If PyOpenMS-Viz was usefull in your research please cite the following:


