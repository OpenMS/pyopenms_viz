from typing import List, Tuple
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, fields
from typing import Literal

from .BasePlotter import _BasePlotter, _BasePlotterConfig, Engine, LegendConfig
from .util._decorators import filter_unexpected_fields

@dataclass(kw_only=True)
class ChromatogramFeatureConfig:
    def default_legend_factory():
        return LegendConfig(title="Features", loc='right', bbox_to_anchor=(1.5, 0.5))

    colormap: str = "viridis"
    lineWidth: float = 1
    lineStyle: str = 'solid'
    legend: LegendConfig = field(default_factory=default_legend_factory)

@filter_unexpected_fields
@dataclass(kw_only=True)
class ChromatogramPlotterConfig(_BasePlotterConfig):
    def default_legend_factory():
        return LegendConfig(title="Transitions")
    
    # Plot Aesthetics
    title: str = "Chromatogram Plot"
    xlabel: str = "Retention Time"
    ylabel: str = "Intensity"
    x_axis_col: str = "rt"
    y_axis_col: str = "int"
    x_axis_location: str = "below"
    y_axis_location: str = "left"
    min_border: int = 0
    show: bool = True
    lineWidth: float = 1
    lineStyle: str = 'solid'
    plot_type: str = "lineplot"
    add_marginals: bool = False
    featureConfig: ChromatogramFeatureConfig = field(default_factory=ChromatogramFeatureConfig)
    legend: LegendConfig = field(default_factory=default_legend_factory)

    # Data Specific Attributes
    ion_mobility: bool = False # if True, plot ion mobility as well in a heatmap


class ChromatogramPlotter(_BasePlotter):
    
    def __init__(self, config: _BasePlotterConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)

    @staticmethod
    def rgb_to_hex(rgb):
        """
        Converts an RGB color value to its corresponding hexadecimal representation.

        Args:
            rgb (tuple): A tuple containing the RGB values as floats between 0 and 1.

        Returns:
            str: The hexadecimal representation of the RGB color.

        Example:
            >>> rgb_to_hex((0.5, 0.75, 1.0))
            '#7fbfff'
        """
        return "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
    

    @staticmethod
    def _get_data_ranges(arr: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float, float, float, float, float, float]:
            """
            Get parameters for plotting.

            Args:
                arr (pd.DataFrame): The input DataFrame.
                
            Returns:
                Tuple[np.ndarray, np.ndarray, float, float, float, float, float, float]: The parameters for plotting.
            """
            im_arr = arr.index.to_numpy()
            rt_arr = arr.columns.to_numpy()

            dw_main = rt_arr.max() - rt_arr.min()
            dh_main = im_arr.max() - im_arr.min()

            rt_min, rt_max, im_min, im_max = rt_arr.min(), rt_arr.max(), im_arr.min(), im_arr.max()

            return im_arr, rt_arr, dw_main, dh_main, rt_min, rt_max, im_min, im_max

    @staticmethod
    def _prepare_array(arr: pd.DataFrame) -> np.ndarray:
            """
            Prepare the array for plotting. Also performs equalization and/or smoothing if specified in the configuration.

            Args:
                arr (pd.DataFrame): The input DataFrame.
                
            Returns:
                np.ndarray: The prepared array.
            """
            arr = arr.to_numpy()
            arr[np.isnan(arr)] = 0
            
            return arr
    
    @staticmethod
    def _get_data_as_two_dimenstional_array(data: pd.DataFrame) -> np.ndarray:
        feat_arrs = {ion_trace:grp_df.pivot_table(index='im', columns='rt', values='int', aggfunc="sum") for ion_trace, grp_df in data.groupby('Annotation')}
        return feat_arrs
    
    @staticmethod
    def _integrate_data_along_dim(data: pd.DataFrame, col_dim: str) -> pd.DataFrame:
        # TODO: Double check which columns are required
        grouped = data.fillna({'native_id': 'NA'}).groupby(['native_id', 'ms_level', 'precursor_mz', 'Annotation', 'product_mz', col_dim])['int'].sum().reset_index()
        return grouped
    
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
        
        def _plotLines(self, data: pd.DataFrame, chromatogramFeatures: DataFrame = None):
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
            p = figure(title=self.config.title, 
                    x_axis_label=self.config.xlabel, 
                    y_axis_label=self.config.ylabel, 
                    x_axis_location=self.config.x_axis_location,
                    y_axis_location=self.config.y_axis_location,
                    width=self.config.width, 
                    height=self.config.height, 
                    tooltips=TOOLTIPS)

            # Create a legend
            legend = Legend()

            # Create a list to store legend items
            if 'Annotation' in data.columns:
                legend_items = []
                i = 0
                for annotation, group_df in data.groupby('Annotation'):
                    source = ColumnDataSource(group_df)
                    line = p.line(x=self.config.x_axis_col, y=self.config.y_axis_col, source=source, line_width=self.config.lineWidth, line_color=self.main_palette[i], line_dash=self.config.lineStyle)
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
                line = p.line(x=self.config.x_axis_col, y=self.config.y_axis_col, source=source, line_width=self.config.lineWidth, line_color=self.main_palette[0], line_alpha=0.5, line_dash=self.config.lineStyle)
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
        
        def _plotHeatmap(self, data: pd.DataFrame):
            from bokeh.plotting import figure, show
            from bokeh.models import ColumnDataSource, Legend, HoverTool, CrosshairTool
            
            AFMHOT_CMAP = [self.rgb_to_hex(cm.afmhot_r(i)[:3]) for i in range(256)]
            
            # Get the data as a two-dimensional array
            feat_arrs = self._get_data_as_two_dimenstional_array(data)
            
            # Get the data ranges
            # Calculate the max - min of 'im' for each group
            im_range = data.groupby('Annotation')['im'].agg(lambda x: x.max() - x.min())
            rt_range = data.groupby('Annotation')['rt'].agg(lambda x: x.max() - x.min())
                    
            p_hm_legends = []
            # Create a legend
            legend_hm = Legend()
            p = figure(x_range=(data.rt.min(), data.rt.max()), 
                    y_range=(data.im.min(), data.im.max()), 
                    x_axis_label="Retention Time [sec]", 
                    y_axis_label="Ion Mobility", 
                    # y_axis_label=None, 
                    # y_axis_location = None,
                    width=700, 
                    height=700, 
                    min_border=0
                    )

            for annotation, df_wide in feat_arrs.items():
                arr = self._prepare_array(df_wide)
                heatmap_img = p.image(image=[arr], x=data.rt.min(), y=data.im.min(), dw=rt_range.mean(), dh=im_range.min(), palette=AFMHOT_CMAP)
                p_hm_legends.append((annotation, [heatmap_img]))

            # Add legend items to the legend
            legend_hm.items = p_hm_legends

            # Add the legend to the plot
            p.add_layout(legend_hm, 'right')

            hover = HoverTool(renderers=[heatmap_img], tooltips=[("Value", "@image")])
            linked_crosshair = CrosshairTool(dimensions="both")
            p.add_tools(hover)
            p.add_tools(linked_crosshair)

            p.grid.visible = False
            
            if self.config.add_marginals:
                # Store original config values that need to be changed for marginals
                show_org = self.config.show
                self.config.show = False
                
                # Integrate the data along the retention time dimension
                rt_integrated = self._integrate_data_along_dim(data, 'rt')
                # Generate a lineplot for XIC
                self.config.y_axis_location = "right"
                p_xic = _plotLines(self, rt_integrated, chromatogramFeatures)
                
                # Link range of XIC plot with the main plot
                p_xic.x_range = p.x_range
                p_xic.width = p.width

                # Modify labels
                p_xic.title = "Integrated Ion Chromatogram"
                # Hide x-axis for grouped plot
                p_xic.xaxis.visible = False

                # Make border 0
                p_xic.min_border = 0

                # Integrate the data along the ion mobility dimension
                im_integrated = self._integrate_data_along_dim(data, 'im')
                
                # Generate a lineplot for XIM
                self.config.x_axis_col = 'int'
                self.config.y_axis_col = 'im'
                self.config.y_axis_location = "left"
                self.config.legend.loc = 'below'
                p_xim = _plotLines(self, im_integrated)
                
                # Link y-axis with heatmap
                p_xim.y_range = p.y_range
                p_xim.height = p.height
                
                p_xim.legend.orientation = "horizontal"

                # Flip x-axis range
                p_xim.x_range.flipped = True
                # p_mobi.y_range

                # Modify labels
                p_xim.title = "Integrated Ion Mobilogram"
                p_xim.title_location = "below"
                p_xim.xaxis.axis_label = "Intensity"
                p_xim.yaxis.axis_label = "Ion Mobility"

                # Make border 0
                p_xim.min_border = 0
                
                # Heatmap mod
                p.yaxis.visible = False
                
                # Construct Marginal Plot
                from bokeh.layouts import gridplot

                # Combine the plots into a grid layout
                layout = gridplot([[None, p_xic], [p_xim, p]], sizing_mode="stretch_both")

                # Reset the config values to org
                self.config.show = show_org
                
                if self.config.show:
                    show(layout)

                return layout
            
            if self.config.show:
                show(p)
                    
            return p

        
        
        if self.config.plot_type == "lineplot":
            return _plotLines(self, data, chromatogramFeatures)
        elif self.config.plot_type == "heatmap":
            return _plotHeatmap(self, data)
        else:
            raise ValueError(f"Invalid plot type: {type}")
  
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
                feature_group = f"Feature {idx}"

                feature_boundary_box = fig.add_shape(type='rect', 
                                                x0=feature['leftWidth'], 
                                                y0=0, 
                                                x1=feature['rightWidth'], 
                                                y1=feature['apexIntensity'],
                                                legendgroup=feature_group,
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
                        legend_labels = [f'Feature {i} (q={feature["q_value"]:.2f})' for i, (_,feature) in enumerate(chromatogramFeatures.iterrows())]
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
                     plot_type: str = "lineplot",
                     add_marginals: bool = False,
                     engine: Literal['PLOTLY', 'BOKEH', 'MATPLOTLIB'] = 'PLOTLY',
                     **kwargs):
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
    if ion_mobility and 'im' in chromatogram.columns:
        x_axis_col = 'int'
        y_axis_col = 'im'
    else:
        x_axis_col = 'rt'
        y_axis_col = 'int'
    config = ChromatogramPlotterConfig(title=title, x_axis_col=x_axis_col, y_axis_col=y_axis_col, show=show_plot, ion_mobility=ion_mobility, width=width, height=height, plot_type=plot_type, add_marginals=add_marginals, engine=engine, **kwargs)
    
    plotter = ChromatogramPlotter(config)
    return plotter.plot(chromatogram, chromatogram_features)

