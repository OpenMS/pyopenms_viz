# MassDash Documentation

*** 

Documentation for the PyOpenMS-Viz project is built using [sphinx](https://www.sphinx-doc.org/en/master/) and is hosted on Read the Docs [MassDash](https://massdash.readthedocs.io/en/latest/).

## Building the Documentation

To build the documentation locally, you will need to install the following dependencies in requirements.txt:

```bash
pip -r requirements.txt
```

Then you can build the documentation using the following command (assuming you are in the docs directory):

```bash
sphinx-build -b html ./source ./build
```

This will build the documentation in the build directory. You can then open the *./build/html/index.html* file in your browser to view the documentation.

Note: The documentation automatically generates the following folders gallery, gallery_template and generated

## Adding Documentation

To add documentation, you will need to add a new **.rst** file in this docs directory. You can then add the file to the *index.rst* file to include it in the documentation.
