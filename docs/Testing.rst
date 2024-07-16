Testing 
=======

Tests are performed using `Pytest <https://docs.pytest.org/en/8.0.x/>`_ and `Syrupy <https://github.com/tophat/syrupy>`_. 

`Syrupy <https://github.com/tophat/syrupy>`_ is used to compare output to previous expected output states. 


SetUp
-----


Install required dependecies using

.. code-block:: bash

    pip install -r requirements-dev.txt

For conformer tests, optional dependencies are also required and can be installed with

.. code-block:: bash

    pip install -r requirements-optional.txt

Running Tests
-------------

To run the tests, execute the following command in the `massdash/` base folder:

.. code-block:: bash

    python -m pytest --snapshot-warn-unused test/

For verbose output:

.. code-block:: bash

    python -m pytest --snapshot-warn-unused -v test/

To update snapshots:

.. code-block:: bash

    python -m pytest --snapshot-update test/


.. note:: CI github testing sometimes fails on macOS, failing to find pytest. If this occurs, please rerun the command.