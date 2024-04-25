from .BasePlotter import _BasePlotter, _BasePlotterConfig
from pandas import DataFrame
from pyopenms import MSChromatogram
from dataclasses import dataclass
from typing import Literal

@dataclass(kw_only=True)
class ChromatogramPlotterConfig(_BasePlotterConfig):
    # Plot Aesthetics
    title: str = "Chromatogram Plot"
    xlabel: str = "Retention Time"
    ylabel: str = "Intensity"
    show_legend: bool = True
    show_plot: bool = True
    
    # Data Specific Attributes
    ion_mobility: bool = False # if True, plot ion mobility as well in a heatmap

class ChromatogramPlotter(_BasePlotter):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def _plotBokeh(self, data: DataFrame):
        from bokeh.plotting import figure, show
        from bokeh.models import ColumnDataSource, Legend
        from bokeh.palettes import Category10

        # Tooltips for interactive information
        TOOLTIPS = [
                ("index", "$index"),
                ("Retention Time", "@rt{0.2f}"),
                ("Intensity", "@int{0.2f}"),
                ("m/z", "@mz{0.4f}")
            ]
        
        if "Annotation" in data.columns:
            TOOLTIPS.append(("Annotation", "@Annotation"))
        else:
            TOOLTIPS.append(("Annotation", "Unknown"))
            data["Annotation"] = "Unknown"
        if "product_mz" in data.columns:
            TOOLTIPS.append(("Target m/z", "@product_mz{0.4f}"))
        
        # Create the Bokeh plot
        p = figure(title="Chromatogram", x_axis_label="Retention Time (min)", y_axis_label="Intensity", plot_width=self.config.width, plot_height=self.config.height, tooltips=TOOLTIPS)

        # Create a legend
        legend = Legend()

        # Create a list to store legend items
        legend_items = []

        colors = Category10[len(data["Annotation"].unique())]
        i = 0
        for annotation, group_df in data.groupby('Annotation'):
            source = ColumnDataSource(group_df)
            line = p.line(x="rt", y="int", source=source, line_width=2, line_color=colors[i], line_alpha=0.5, line_dash='solid')
            legend_items.append((annotation, [line]))
            i+=1

        # Add legend items to the legend
        legend.items = legend_items

        # Add the legend to the plot
        p.add_layout(legend, 'right')

        p.legend.location = "top_left"
        # p.legend.click_policy="hide"
        p.legend.click_policy="mute"
        p.legend.title = "Transition"
        p.legend.label_text_font_size = "10pt"

        # Customize the plot
        p.grid.visible = True
        p.toolbar_location = "above"
        
        if self.config.show_plot:
            show(p)
        else:
            return p

    def _plotPlotly(self, chromatogram: MSChromatogram):
        ##TODO
        pass  


# ============================================================================= #
## FUNCTIONAL API ##
# ============================================================================= #

def plotChromatogram(chromatogram: MSChromatogram, 
                     title: str = "Chromatogram Plot",
                     ion_mobility: bool = False,
                     width: int = 500,
                     height: int = 500,
                     engine: Literal['PLOTLY', 'BOKEH'] = 'PLOTLY'):
    """
    Plot a Chromatogram from a MSChromatogram Object

    Args:
        chromatogram (MSChromatogram): OpenMS chromatogram object
        title (str, optional): title of plot. Defaults to "Chromatogram Plot".
        ion_mobility (bool, optional): If True, plots a heatmap of Retention Time vs ion mobility with intensity as the color. Defaults to False.
        width (int, optional): width of the figure. Defaults to 500.
        height (int, optional): height of the figure. Defaults to 500.
        engine (Literal['PLOTLY', 'BOKEH'], optional): Plotting engine to use. Defaults to 'PLOTLY'. Can be either 'PLOTLY' or 'BOKEH'
    
    Returns:
        PLOTLY figure or BOKEH figure depending on engine
    """

    config = ChromatogramPlotterConfig(title=title, ion_mobility=ion_mobility, width=width, height=height, engine=engine)
    plotter = ChromatogramPlotter(config)
    return plotter.plot(chromatogram)

