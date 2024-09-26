Installation
============

(Recommended) The Python Package Index 
--------------------------------------

The recommended way of installing pyOpenMS-viz is through the Python Package Index (PyPI). We recommend installing MassDash in its own virtual environment using Anaconda to avoid packaging conflicts.

First create a new environment:

.. code-block:: bash

   conda create --name=pyopenms-viz python=3.11
   conda activate pyopenms-viz

Then in the new environment install pyOpenMS-viz.

.. code-block:: bash

   pip install pyopenmsviz --upgrade

After installation the GUI can be launched in the Terminal/Anaconda Prompt using 

Building from Source
--------------------

The source code is freely open and accessible on Github at https://github.com/OpenMS/pyopenms_viz under the `BSD-3-Clause license <https://github.com/OpenMS/pyopenms_viz/blob/main/LICENSE>`_ The package can be installed by cloning and installing from source using pip.

First clone the repository:

.. code-block:: bash

        git clone git@github.com:OpenMS/pyopenms-viz.git

Change into the pyopenms-viz directory

.. code-block:: bash
        
        cd pyopenms-viz

Install using pip

.. code-block:: bash

        pip install -e .
