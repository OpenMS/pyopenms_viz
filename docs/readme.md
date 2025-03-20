# Documentation

## Building Docs Locally

1. Create a new conda environment based on the docs_env.yml file
	`conda env create -f docs_env.yml`

	This environment contains requirements.txt jupyter-lab pandoc, cairo_svg and rdkit

2. Build docs with sphinx-build
	`sphinx-build -b html ./ ./_build` ensure this is executed in the docs folder
