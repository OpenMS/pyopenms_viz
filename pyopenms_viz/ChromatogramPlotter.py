from .BasePlotter import _BasePlotter, _BasePlotterConfig, Engine
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
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
    def __init__(self, config: _BasePlotterConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        
    @staticmethod
    def generate_colors(n):
        # Use Matplotlib's built-in color palettes
        cmap = plt.get_cmap('tab20', n)
        colors = cmap(np.linspace(0, 1, n))
        
        # Convert colors to hex format
        hex_colors = ['#{:02X}{:02X}{:02X}'.format(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors]
        
        return hex_colors
    
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
            TOOLTIPS.append(("Annotation", "@Annotation"))
            data = data.copy()
            data.loc[:, "Annotation"] = "Unknown"
        if "product_mz" in data.columns:
            TOOLTIPS.append(("Target m/z", "@product_mz{0.4f}"))
        
        # Create the Bokeh plot
        p = figure(title=self.config.title, x_axis_label=self.config.xlabel, y_axis_label=self.config.ylabel, width=self.config.width, height=self.config.height, tooltips=TOOLTIPS)

        # Create a legend
        legend = Legend()

        # Create a list to store legend items
        legend_items = []
        colors = self.generate_colors(len(data["Annotation"].unique()))
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

    def _plotPlotly(self, data: DataFrame):
        import plotly.graph_objects as go

        if "Annotation" not in data.columns:
            data = data.copy()
            data.loc[:, "Annotation"] = "Unknown"

        # Create a trace for each unique annotation
        traces = []
        colors = self.generate_colors(len(data["Annotation"].unique()))
        for i, (annotation, group_df) in enumerate(data.groupby('Annotation')):
            trace = go.Scatter(
                x=group_df["rt"],
                y=group_df["int"],
                mode='lines',
                name=annotation,
                line=dict(
                    color=colors[i],
                    width=2,
                    dash='solid'
                )
            )
            traces.append(trace)

        # Create the Plotly figure
        fig = go.Figure(data=traces)
        fig.update_layout(
            title=self.config.title,
            xaxis_title=self.config.xlabel,
            yaxis_title=self.config.ylabel,
            width=self.config.width,
            height=self.config.height,
            legend_title="Transition",
            legend_font_size=10
        )

        # Add tooltips
        fig.update_traces(
            hovertemplate=(
                "Index: %{customdata[0]}<br>" +
                "Retention Time: %{x:.2f}<br>" +
                "Intensity: %{y:.2f}<br>" +
                "m/z: %{customdata[1]:.4f}<br>" +
                "Annotation: %{customdata[2]}"
            ),
            customdata=list(zip(data.index, data["mz"], data["Annotation"]))
        )

        # Customize the plot
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_showgrid=True,
            yaxis_showgrid=True,
            xaxis_zeroline=False,
            yaxis_zeroline=False
        )

        if self.config.show_plot:
            fig.show()
        else:
            return fig 
    
    def plot(self, data, **kwargs):
        if self.config.engine_enum == Engine.PLOTLY:
            return self._plotPlotly(data, **kwargs)
        else: # self.config.engine_enum == Engine.BOKEH:
            return self._plotBokeh(data, **kwargs)


# ============================================================================= #
## FUNCTIONAL API ##
# ============================================================================= #

def plotChromatogram(chromatogram: MSChromatogram, 
                     title: str = "Chromatogram Plot",
                     show_legend: bool = True,
                     show_plot: bool = True,
                     ion_mobility: bool = False,
                     width: int = 500,
                     height: int = 500,
                     engine: Literal['PLOTLY', 'BOKEH'] = 'PLOTLY'):
    """
    Plot a Chromatogram from a MSChromatogram Object

    Args:
        chromatogram (MSChromatogram): OpenMS chromatogram object
        title (str, optional): title of plot. Defaults to "Chromatogram Plot".
        show_legend (bool, optional): If True, shows the legend. Defaults to True.
        show_plot (bool, optional): If True, shows the plot. Defaults to True.
        ion_mobility (bool, optional): If True, plots a heatmap of Retention Time vs ion mobility with intensity as the color. Defaults to False.
        width (int, optional): width of the figure. Defaults to 500.
        height (int, optional): height of the figure. Defaults to 500.
        engine (Literal['PLOTLY', 'BOKEH'], optional): Plotting engine to use. Defaults to 'PLOTLY'. Can be either 'PLOTLY' or 'BOKEH'
    
    Returns:
        PLOTLY figure or BOKEH figure depending on engine
    """

    config = ChromatogramPlotterConfig(title=title, show_legend=show_legend, show_plot=show_plot, ion_mobility=ion_mobility, width=width, height=height, engine=engine)
    plotter = ChromatogramPlotter(config)
    return plotter.plot(chromatogram)

