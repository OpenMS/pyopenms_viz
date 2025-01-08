Peak Map
========

Peak Maps can be plotted using kind = peakmap. Commonly in this plot, mass-to-charge is on the x-axis, retention time is on the y-axis and intensity is on the z-axis (or represented by color). The x and y axis can be changed based on use case, for example y can also be ion mobility. Using `plot_3d=True` enables 3D plotting. Currently 3D plotting only supported for `ms_matplotlib` and `ms_plotly` backends.



Parameters
----------

.. csv-table:: 
   :file: peakMapPlot.tsv
   :header-rows: 1
   :delim: tab


Example Usage
-------------


.. minigallery::

   gallery_scripts/ms_bokeh/plot_peakmap_marginals_ms_bokeh.py  
   gallery_scripts/ms_bokeh/plot_peakmap_ms_bokeh.py
   gallery_scripts/ms_matplotlib/plot_peakmap_ms_matplotlib.py
   gallery_scripts/ms_matplotlib/plot_peakmap_ms_matplotlib.py
   gallery_scripts/ms_plotly/plot_peakmap_ms_plotly.py
   gallery_scripts/ms_plotly/plot_peakmap_ms_plotly.py
   gallery_scripts/ms_matplotlib/plot_peakmap_3D_ms_matplotlib.py
   gallery_scripts/ms_plotly/plot_peakmap_3D_ms_plotly.py
   gallery_scripts/ms_matplotlib/plot_peakmap_3D_highlight_peptide_ms_matplotlib.py
   gallery_scripts/ms_plotly/plot_peakmap_3D_highlight_peptide_ms_plotly.py





