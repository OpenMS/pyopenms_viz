Installation
============

(Recommended) The Python Package Index 
--------------------------------------

The recommended way of installing MassDash is through the Python Package Index (PyPI). We recommend installing MassDash in its own virtual environment using Anaconda to avoid packaging conflicts.

First create a new environment:

.. code-block:: bash

   conda create --name=massdash python=3.9
   conda activate massdash 

Then in the new environment install MassDash.

.. code-block:: bash

   pip install pyopenmsviz --upgrade

After installation the GUI can be launched in the Terminal/Anaconda Prompt using 

Building from Source
--------------------

The source code is freely open and accessible on Github at https://github.com/Roestlab/massdash under the `BSD-3-Clause license <https://github.com/Roestlab/massdash?tab=BSD-3-Clause-1-ov-file>`_. The package can be installed by cloning and installing from source using pip.

First clone the repository:

.. code-block:: bash

        git clone git@github.com:jcharkow/pyopenms-viz.git

Change into the massdash directory

.. code-block:: bash
        
        cd pyopenms-viz

Install using pip

.. code-block:: bash

        pip install -e .
