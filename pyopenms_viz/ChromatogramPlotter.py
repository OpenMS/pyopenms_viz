from pyopenms_viz.BasePlotter import _BasePlotter, _BasePlotterConfig, Engine, LegendConfig
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal

@dataclass(kw_only=True)
class ChromatogramFeatureConfig:
    def default_legend_factory():
        return LegendConfig(title="Features", loc='right', bbox_to_anchor=(1.5, 0.5))

    colormap: str = "viridis"
    lineWidth: float = 1
    lineStyle: str = 'solid'
    legend: LegendConfig = field(default_factory=default_legend_factory)

@dataclass(kw_only=True)
class ChromatogramPlotterConfig(_BasePlotterConfig):
    def default_legend_factory():
        return LegendConfig(title="Transitions")
    
    # Plot Aesthetics
    title: str = "Chromatogram Plot"
    xlabel: str = "Retention Time"
    ylabel: str = "Intensity"
    show: bool = True
    lineWidth: float = 1
    lineStyle: str = 'solid'
    featureConfig: ChromatogramFeatureConfig = field(default_factory=ChromatogramFeatureConfig)
    legend: LegendConfig = field(default_factory=default_legend_factory)

    # Data Specific Attributes
    ion_mobility: bool = False # if True, plot ion mobility as well in a heatmap


class ChromatogramPlotter(_BasePlotter):
    def __init__(self, config: _BasePlotterConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)

    ### assume that the chromatogram have the following columns: intensity, time 
    ### optional column to be used: annotation
    ### assume that the chromatogramFeatures have the following columns: left_width, right_width (optional columns: area, q_value)
    def plot(self, chromatogram, chromatogramFeatures = None, **kwargs):
        #### General Data Processing before plotting ####
        # sort by q_value if available
        if chromatogramFeatures is not None:
            if "q_value" in chromatogramFeatures.columns:
                chromatogramFeatures = chromatogramFeatures.sort_values(by="q_value")

        #### compute apex intensity for features if not already computed 
        if chromatogramFeatures is not None:
            if "apexIntensity" not in chromatogramFeatures.columns:
                all_apexIntensity = []
                for _, feature in chromatogramFeatures.iterrows():
                    apexIntensity = 0
                    for _, row in chromatogram.iterrows():
                        if row["rt"] >= feature["leftWidth"] and row["rt"] <= feature["rightWidth"] and row["int"] > apexIntensity:
                            apexIntensity = row["int"]
                    all_apexIntensity.append(apexIntensity)

                chromatogramFeatures["apexIntensity"] = all_apexIntensity

        # compute colormaps based on the number of transitions and features
        self.main_palette = self.generate_colors(self.config.colormap, len(chromatogram["Annotation"].unique()) if 'Annotation' in chromatogram.columns else 1)
        self.feature_palette = self.generate_colors(self.config.featureConfig.colormap, len(chromatogramFeatures)) if chromatogramFeatures is not None else None

        return super().plot(chromatogram, chromatogramFeatures, **kwargs)

    def _plotBokeh(self, data: DataFrame, chromatogramFeatures: DataFrame = None):
        from bokeh.plotting import figure, show
        from bokeh.models import ColumnDataSource, Legend
        
        # Tooltips for interactive information
        TOOLTIPS = [
                ("index", "$index"),
                ("Retention Time", "@rt{0.2f}"),
                ("Intensity", "@int{0.2f}"),
                ("m/z", "@mz{0.4f}")
            ]
        
        if "Annotation" in data.columns:
            TOOLTIPS.append(("Annotation", "@Annotation"))
        if "product_mz" in data.columns:
            TOOLTIPS.append(("Target m/z", "@product_mz{0.4f}"))
        
        # Create the Bokeh plot
        p = figure(title=self.config.title, x_axis_label=self.config.xlabel, y_axis_label=self.config.ylabel, width=self.config.width, height=self.config.height, tooltips=TOOLTIPS)

        # Create a legend
        legend = Legend()

        # Create a list to store legend items
        if 'Annotation' in data.columns:
            legend_items = []
            i = 0
            for annotation, group_df in data.groupby('Annotation'):
                source = ColumnDataSource(group_df)
                line = p.line(x="rt", y="int", source=source, line_width=self.config.lineWidth, line_color=self.main_palette[i], line_dash=self.config.lineStyle)
                legend_items.append((annotation, [line]))
                i+=1
                
            # Add legend items to the legend
            legend.items = legend_items

            # Add the legend to the plot
            p.add_layout(legend, self.config.legend.loc)

            p.legend.click_policy=self.config.legend.onClick
            p.legend.title = self.config.legend.title
            p.legend.label_text_font_size = str(self.config.legend.fontsize) + 'pt'

        else:
            source = ColumnDataSource(data)
            line = p.line(x="rt", y="int", source=source, line_width=self.config.lineWidth, line_color=self.main_palette[0], line_alpha=0.5, line_dash=self.config.lineStyle)
        # Customize the plot
        p.grid.visible = self.config.grid
        p.toolbar_location = "above" #NOTE: This is hardcoded
       
        ##### Plotting chromatogram features #####
        if chromatogramFeatures is not None:

            for idx, (_, feature) in enumerate(chromatogramFeatures.iterrows()):

                leftWidth_line = p.line(x=[feature['leftWidth']] * 2, y=[0, feature['apexIntensity']], width=self.config.featureConfig.lineWidth, color=self.feature_palette[idx], line_dash=self.config.featureConfig.lineStyle )
                rightWidth_line = p.line(x=[feature['rightWidth']] * 2, y=[0, feature['apexIntensity']], width=self.config.featureConfig.lineWidth, color=self.feature_palette[idx], line_dash = self.config.featureConfig.lineStyle)

                if self.config.featureConfig.legend.show:
                    feature_legend_items = []
                    if "q_value" in chromatogramFeatures.columns:
                        legend_msg = f'Feature {idx} (q={feature["q_value"]:.2f})'
                    else:
                        legend_msg = f'Feature {idx}'
                    feature_legend_items.append((legend_msg, [leftWidth_line]))

                    legend = Legend(items=feature_legend_items, title=self.config.legend.title )
                    p.add_layout(legend, self.config.legend.loc)

        if self.config.show:
            show(p)

        return p
  
    def _plotPlotly(self, data: DataFrame, chromatogramFeatures: DataFrame):
        import plotly.graph_objects as go

        # Create a trace for each unique annotation
        traces = []
        if "Annotation" in data.columns:
            for i, (annotation, group_df) in enumerate(data.groupby('Annotation')):
                trace = go.Scatter(
                    x=group_df["rt"],
                    y=group_df["int"],
                    mode='lines',
                    name=annotation,
                    line=dict(
                        color=self.main_palette[i],
                        width=self.config.lineWidth,
                        dash=self.config.lineStyle
                    )
                )
                traces.append(trace)
        else:
            trace = go.Scatter(
                x=data["rt"],
                y=data["int"],
                mode='lines',
                name="Transition",
                line=dict(
                    color=self.main_palette[0],
                    width=self.config.lineWidth,
                    dash=self.config.lineStyle
                ))
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
            legend_font_size=self.config.legend.fontsize
        )

        if "Annotation" in data.columns:
            # Add tooltips
            fig.update_traces(
                hovertemplate=(
                    "Index: %{customdata[0]}<br>" +
                    "Retention Time: %{x:.2f}<br>" +
                    "Intensity: %{y:.2f}<br>" +
                    "m/z: %{customdata[1]:.4f}<br>" +
                    "Annotation: %{customdata[2]}"
                ),
                customdata=list(zip(data.index, data["mz"], data["Annotation"])))
        else:
            # Add tooltips
            fig.update_traces( hovertemplate=( 
                    "Index: %{customdata[0]}<br>" +
                    "Retention Time: %{x:.2f}<br>" +
                    "Intensity: %{y:.2f}<br>" +
                    "m/z: %{customdata[1]:.4f}<br>"
                ),
                customdata=list(zip(data.index, data["mz"])))

        # Customize the plot
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_showgrid=True,
            yaxis_showgrid=True,
            xaxis_zeroline=False,
            yaxis_zeroline=False
        )

        ##### Plotting chromatogram features #####
        if chromatogramFeatures is not None:
            for idx, (_, feature) in enumerate(chromatogramFeatures.iterrows()):

                leftWidth_line = fig.add_shape(type='line', 
                                               x0=feature['leftWidth'], 
                                               y0=0, 
                                               x1=feature['leftWidth'], 
                                               y1=feature['apexIntensity'], 
                                               line=dict(
                                                   color=self.feature_palette[idx],
                                                   width=self.config.featureConfig.lineWidth,
                                                    dash=self.config.featureConfig.lineStyle)
                )

                rightWidth_line = fig.add_shape(type='line', 
                                                x0=feature['rightWidth'], 
                                                y0=0, 
                                                x1=feature['rightWidth'], 
                                                y1=feature['apexIntensity'],
                                                legendgroup="features",
                                                legendgrouptitle_text="Features",
                                                showlegend=self.config.featureConfig.legend.show,
                                                name=f'Feature {idx}' if "q_value" not in chromatogramFeatures.columns else f'Feature {idx} (q={feature["q_value"]:.2f})',
                                                line=dict(
                                                   color=self.feature_palette[idx],
                                                   width=self.config.featureConfig.lineWidth,
                                                   dash=self.config.featureConfig.lineStyle)
                )

        if self.config.show:
            fig.show()
        return fig 
        
    def _plotMatplotlib(self, data: DataFrame, chromatogramFeatures: DataFrame = None):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(self.config.width/100, self.config.height/100), dpi=100)

        # Set plot title and axis labels
        ax.set_title(self.config.title)
        ax.set_xlabel(self.config.xlabel)
        ax.set_ylabel(self.config.ylabel)

        # Create a legend
        legend_lines = []
        legend_labels = []

        # Plot each unique annotation
        if "Annotation" in data.columns:
            for i, (annotation, group_df) in enumerate(data.groupby('Annotation')):
                line, = ax.plot(group_df["rt"], group_df["int"], color=self.main_palette[i], linewidth=self.config.lineWidth, ls=self.config.lineStyle)
                legend_lines.append(line)
                legend_labels.append(annotation)

            # Add legend
            matplotlibLegendLoc= LegendConfig._matplotlibLegendLocationMapper(self.config.legend.loc)
            legend = ax.legend(legend_lines, legend_labels, loc=matplotlibLegendLoc, bbox_to_anchor=self.config.legend.bbox_to_anchor, title=self.config.legend.title, prop={'size': self.config.legend.fontsize})
            legend.get_title().set_fontsize(str(self.config.legend.fontsize))

        else: # only one transition
            line, = ax.plot(data["rt"], data["int"], color=self.main_palette[0], linewidth=self.config.lineWidth, ls=self.config.lineStyle)

        # Customize the plot
        ax.grid(self.config.grid)

        ## add 10% padding to the plot
        padding = (data['int'].max() - data['int'].min() ) * 0.1
        ax.set_xlim(data["rt"].min(), data["rt"].max())
        ax.set_ylim(data["int"].min(), data["int"].max() + padding)

        ##### Plotting chromatogram features #####
        if chromatogramFeatures is not None:
            ax.add_artist(legend)

            for idx, (_, feature) in enumerate(chromatogramFeatures.iterrows()):

                ax.vlines(x=feature['leftWidth'], ymin=0, ymax=feature['apexIntensity'], lw=self.config.featureConfig.lineWidth, color=self.feature_palette[idx], ls=self.config.featureConfig.lineStyle)
                ax.vlines(x=feature['rightWidth'], ymin=0, ymax=feature['apexIntensity'], lw=self.config.featureConfig.lineWidth, color=self.feature_palette[idx], ls=self.config.featureConfig.lineStyle)

                if self.config.featureConfig.legend.show:
                    custom_lines = [Line2D([0], [0], color=self.feature_palette[i], lw=2) for i in range(len(chromatogramFeatures))]
                    if "q_value" in chromatogramFeatures.columns:
                        legend_labels = [f'Feature {i} (q={feature["q_value"]:.2f})' for i, feature in enumerate(chromatogramFeatures.iterrows())]
                    else:
                        legend_labels = [f'Feature {i}' for i in range(len(chromatogramFeatures))]

            if self.config.featureConfig.legend.show:

                matplotlibLegendLoc= LegendConfig._matplotlibLegendLocationMapper(self.config.featureConfig.legend.loc)
                ax.legend(custom_lines, legend_labels, loc=matplotlibLegendLoc, bbox_to_anchor=self.config.featureConfig.legend.bbox_to_anchor, title=self.config.featureConfig.legend.title)

        if self.config.show:
            plt.show()
        return fig

# ============================================================================= #
## FUNCTIONAL API ##
# ============================================================================= #
def plotChromatogram(chromatogram: pd.DataFrame, 
                     chromatogram_features: pd.DataFrame = None,
                     title: str = "Chromatogram Plot",
                     show_plot: bool = True,
                     ion_mobility: bool = False,
                     width: int = 500,
                     height: int = 500,
                     engine: Literal['PLOTLY', 'BOKEH', 'MATPLOTLIB'] = 'PLOTLY'):
    """
    Plot a Chromatogram from a MSChromatogram Object

    Args:
        chromatogram (DataFrame): DataFrame containing chromatogram data 
        chromatogram_features (DataFrame, optional): DataFrame containing chromatogram features. Defaults to None.
        title (str, optional): title of plot. Defaults to "Chromatogram Plot".
        show_plot (bool, optional): If True, shows the plot. Defaults to True.
        ion_mobility (bool, optional): If True, plots a heatmap of Retention Time vs ion mobility with intensity as the color. Defaults to False.
        width (int, optional): width of the figure. Defaults to 500.
        height (int, optional): height of the figure. Defaults to 500.
        engine (Literal['PLOTLY', 'BOKEH'], optional): Plotting engine to use. Defaults to 'PLOTLY'. Can be either 'PLOTLY' or 'BOKEH'
    
    Returns:
        PLOTLY figure or BOKEH figure depending on engine
    """

    config = ChromatogramPlotterConfig(title=title, show=show_plot, ion_mobility=ion_mobility, width=width, height=height, engine=engine)
    plotter = ChromatogramPlotter(config)
    return plotter.plot(chromatogram, chromatogram_features)

