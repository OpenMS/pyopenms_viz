from __future__ import annotations

from abc import ABC

from typing import TYPE_CHECKING, Literal, List, Tuple, Union

import plotly.graph_objects as go
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

from pandas.core.frame import DataFrame

from numpy import column_stack

from .._core import BasePlotter, LinePlot, VLinePlot, ScatterPlot, ComplexPlot, ChromatogramPlot, MobilogramPlot, SpectrumPlot, FeatureHeatmapPlot

from pyopenms_viz.plotting._misc import ColorGenerator
from pyopenms_viz.constants import PEAK_BOUNDARY_ICON, FEATURE_BOUNDARY_ICON

class PLOTLYPlot(BasePlotter, ABC):
    """
    Base class for assembling a Ploty plot
    """

    @property
    def _interactive(self) -> bool:
        return True

    def _load_extension(self):
        '''
        Tries to load the plotly extensions, if not throw an import error
        '''
        try:
            import plotly.graph_objects
        except ImportError:
            raise ImportError(
                f"plotly is not installed. Please install using `pip install plotly` to use this plotting library in pyopenms-viz"
            )
        
    def _create_figure(self):
        '''
        Create a new figure, if a figure is not supplied
        '''
        if self.fig is None:
            self.fig = go.Figure()
            self.fig.update_layout(
                title=self.title,
                xaxis_title=self.xlabel,
                yaxis_title=self.ylabel,
                width=self.width,
                height=self.height
            )
 
    def _update_plot_aes(self, fig, **kwargs) -> None:
        """
        Update the plot aesthetics.
        """
        fig.update_layout(
            legend_title = self.config.legend.title,
            legend_font_size = self.config.legend.fontsize,
            showlegend = self.config.legend.show
        )
        
        # Update to look similar to Bokeh theme
        # Customize the layout
        fig.update_layout(
            plot_bgcolor='#FFFFFF',  # Set the plot background color
            font_family='Helvetica',  # Set the font family
            font_size=12,  # Set the font size
            title_font_family='Helvetica',  # Set the title font family
            title_font_size=16,  # Set the title font size
            xaxis_title_font_family='Helvetica',  # Set the x-axis title font family
            xaxis_title_font_size=14,  # Set the x-axis title font size
            yaxis_title_font_family='Helvetica',  # Set the y-axis title font family
            yaxis_title_font_size=14,  # Set the y-axis title font size
            xaxis_gridcolor='#CCCCCC',  # Set the x-axis grid color
            yaxis_gridcolor='#CCCCCC',  # Set the y-axis grid color
            xaxis_tickfont_family='Helvetica',  # Set the x-axis tick font family
            yaxis_tickfont_family='Helvetica',  # Set the y-axis tick font family
            legend_font_family='Helvetica',  # Set the legend font family
        )
        
        # Add x-axis grid lines and ticks
        fig.update_xaxes(
            showgrid=self.grid,  # Add x-axis grid lines
            showline=True, 
            linewidth=1, 
            linecolor='black',
            ticks='outside',  # Add x-axis ticks outside the plot area
            tickwidth=1,  # Set the width of x-axis ticks
            tickcolor='black',  # Set the color of x-axis ticks
        )

        # Add y-axis grid lines and ticks
        fig.update_yaxes(
            showgrid=self.grid,  # Add y-axis grid lines
            showline=True, 
            linewidth=1, 
            linecolor='black',
            tickwidth=1,  # Set the width of y-axis ticks
            tickcolor='black'  # Set the color of y-axis ticks
        )
            
    def _add_legend(self, fig, legend):
        pass
    
    def _add_tooltips(self, fig, tooltips, custom_hover_data=None):
        fig.update_traces(
            hovertemplate=tooltips,
            customdata=custom_hover_data
        )
    
    def _add_bounding_box_drawer(self, fig, **kwargs):
        fig.update_layout(modebar_add=['drawrect',
                                        'eraseshape'
                                       ],
                          newshape=dict(
                                showlegend=True,
                                label=dict(
                                           texttemplate="x0: %{x0:.2f} | x1: %{x1:.2f}<br>y0: %{y0:.2f} | y1: %{y1:.2f}",
                                           textposition="top left",),
                                line_color="#F02D1A",
                                fillcolor=None,
                                line=dict(
                                    dash="dash",
                                ),
                                opacity=0.5
                            ))
    
    def _add_bounding_vertical_drawer(self, fig, **kwargs):
        # Note: self.label_suffix must be defined
        
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines")
        )
        fig.update_layout(modebar_add=['drawrect',
                                        'eraseshape'
                                       ],
                          newshape=dict(
                                showlegend=True,
                                label=dict(
                                           texttemplate=self.label_suffix + "_0: %{x0:.2f} | " + self.label_suffix + "_1: %{x1:.2f}",
                                           textposition="top left",),
                                line_color="#F02D1A",
                                fillcolor=None,
                                line=dict(
                                    dash="dash",
                                ),
                                drawdirection="vertical",
                                opacity=0.5
                            )
                          )
    
    def _modify_x_range(self, x_range: Tuple[float, float] | None = None, padding: Tuple[float, float] | None = None):
        start, end = x_range
        if padding is not None:
            start = start - (start * padding[0])
            end = end + (end * padding[1])
        self.fig.update_xaxes(range=[start, end])
    
    def _modify_y_range(self, y_range: Tuple[float, float] | None = None, padding: Tuple[float, float] | None = None):
        start, end = y_range
        if padding is not None:
            start = start - (start * padding[0])
            end = end + (end * padding[1])
        self.fig.update_yaxes(range=[start, end])
    
    def show(self, **kwargs):
        self.fig.show(**kwargs)

class PLOTLYLinePlot(PLOTLYPlot, LinePlot):
    """
    Class for assembling a set of line plots in plotly
    """
    
    @classmethod
    def plot( # type: ignore[override]
        cls,
        fig,
        data: DataFrame,
        x: Union[str, int],
        y: Union[str, int],
        by: str | None = None,
        **kwargs
    ) -> Tuple[Figure, "Legend"]: # note legend is always none for consistency
        color_gen = kwargs.pop("line_color", None)
        
        traces = []
        if by is None:
            trace = go.Scatter(
                x=data[x],
                y=data[y],
                mode="lines",
                line=dict(
                    color=next(color_gen)
                    )
            )
            traces.append(trace)
        else:
            for group, df in data.groupby(by):
                trace = go.Scatter(
                    x=df[x],
                    y=df[y],
                    mode="lines",
                    name=group,
                    line=dict(
                        color=next(color_gen)
                    )
                )
                traces.append(trace)
                

        fig.add_traces(data=traces)
        return fig, None


class PLOTLYVLinePlot(PLOTLYPlot, VLinePlot):
           
    @classmethod
    def plot(cls, fig, data, x, y, by=None, **kwargs) -> Tuple[Figure, "Legend"]:
        color_gen = kwargs.pop("line_color", None)
        
        traces = []
        if by is None:
            line_color = next(color_gen)
            if "showlegend" in kwargs:
                showlegend = kwargs["showlegend"]
                first_group_trace_showlenged = showlegend
            else:
                first_group_trace_showlenged = True
            for _, row in data.iterrows():
                trace = go.Scattergl(
                    x=[row[x]] * 2,
                    y=[0, row[y]],
                    mode="lines",
                    name="Trace",
                    legendgroup="Trace",
                    showlegend=first_group_trace_showlenged,
                    line=dict(
                        color=line_color
                        )
                )
                first_group_trace_showlenged = False
                traces.append(trace)
        else:
            for group, df in data.groupby(by):
                line_color = next(color_gen)
                if "showlegend" in kwargs:
                    showlegend = kwargs["showlegend"]
                    first_group_trace_showlenged = showlegend
                else:
                    first_group_trace_showlenged = True
                for _, row in df.iterrows():
                    trace = go.Scattergl(
                        x=[row[x]] * 2,
                        y=[0, row[y]],
                        mode="lines",
                        name=group,
                        legendgroup=group,
                        showlegend=first_group_trace_showlenged,
                        line=dict(
                            color=line_color
                        )
                    )
                    first_group_trace_showlenged = False
                    traces.append(trace)
                
        fig.add_traces(data=traces)
        return fig, None


class PLOTLYScatterPlot(PLOTLYPlot, ScatterPlot):
           
    @classmethod
    def plot(cls, fig, data, x, y, by=None, **kwargs) -> Tuple[Figure, "Legend"]:
        color_gen = kwargs.pop("line_color", None)
        marker_dict = kwargs.pop("marker", None)
        
        if color_gen is None:
            color_gen = ColorGenerator()
        traces = []
        if by is None:
            if marker_dict is None:
                marker_dict = dict(color=next(color_gen))
            trace = go.Scattergl(
                x=data[x],
                y=data[y],
                mode="markers",
                marker=dict(
                    color=next(color_gen)
                )
            )
            traces.append(trace)
        else:
            for group, df in data.groupby(by):
                if marker_dict is None:
                    marker_dict = dict(color=next(color_gen))
                    
                trace = go.Scatter(
                    x=df[x],
                    y=df[y],
                    mode="markers",
                    name=group,
                    marker=marker_dict, 
                    **kwargs
                )
                traces.append(trace)
                
        fig.add_traces(data=traces)
        return fig, None

class PLOTLYComplexPlot(ComplexPlot, PLOTLYPlot, ABC):

    def get_line_renderer(self, data, x, y, **kwargs) -> None:
        return PLOTLYLinePlot(data, x, y, **kwargs)
    
    def get_vline_renderer(self, data, x, y, **kwargs) -> None:
        return PLOTLYVLinePlot(data, x, y, **kwargs)
    
    def get_scatter_renderer(self, data, x, y, **kwargs) -> None:
        return PLOTLYScatterPlot(data, x, y, **kwargs)
    
    def plot_x_axis_line(self, fig):
        fig.add_hline(y=0, line_color="black", line=dict(width=1))
    
    def _create_tooltips(self):
        available_columns = self.data.columns.tolist()
        # Get index (not a column in dataframe) data first for customer hover data
        custom_hover_data = [self.data.index]
        # Get the rest of the columns
        custom_hover_data  += [self.data[col] for col in ["mz", "Annotation", "product_mz"] if col in available_columns]
        
        TOOLTIPS = [
            "Index: %{customdata[0]}",
            "Retention Time: %{x:.2f}",
            "Intensity: %{y:.2f}",
        ]
        
        custom_hover_data_index = 1
        if "mz" in self.data.columns:
            TOOLTIPS.append("m/z: %{customdata[" + str(custom_hover_data_index) + "]:.4f}")
            custom_hover_data_index += 1
        if "Annotation" in self.data.columns:
            TOOLTIPS.append("Annotation: %{customdata[" + str(custom_hover_data_index) + "]}")
            custom_hover_data_index += 1
        if "product_mz" in self.data.columns:
            TOOLTIPS.append("Target m/z: %{customdata[" + str(custom_hover_data_index) + "]:.4f}")
        
        return "<br>".join(TOOLTIPS), column_stack(custom_hover_data)

class PLOTLYChromatogramPlot(PLOTLYComplexPlot, ChromatogramPlot):

    def _add_peak_boundaries(self, feature_data, **kwargs):
        color_gen = ColorGenerator(
            colormap=self.config.feature_config.colormap, n=feature_data.shape[0]
        )
        for idx, (_, feature) in enumerate(feature_data.iterrows()):
            if "q_value" in feature_data.columns:
                legend_label = f"Feature {idx} (q-value: {feature['q_value']:.4f})"
            else:
                legend_label = f"Feature {idx}"
            self.fig.add_trace(go.Scatter(
                mode="lines",
                x=[feature["leftWidth"], feature["leftWidth"], feature["rightWidth"], feature["rightWidth"]],
                y=[0, feature["apexIntensity"], 0, feature["apexIntensity"]],
                fillcolor=next(color_gen),
                opacity=0.5,
                line=dict(
                    dash="dash",
                    width=2.5
                ),
                name=legend_label
            )
            )
    
    def get_manual_bounding_box_coords(self, arg):
        # TODO: Implement this method, plotly doesn't have a direct easy way of extracting the relayout events. Would need to implement / add a dash dependency to add a callback to extract the relayout events
        pass


class PLOTLYMobilogramPlot(PLOTLYChromatogramPlot, MobilogramPlot):
    pass


class PLOTLYSpectrumPlot(PLOTLYComplexPlot, SpectrumPlot):
    pass
    
class PLOTLYFeatureHeatmapPlot(PLOTLYComplexPlot, FeatureHeatmapPlot):

    def create_main_plot(self, x, y, z, class_kwargs, other_kwargs):
        scatterPlot = self.get_scatter_renderer(self.data, x, y, **class_kwargs)
        self.fig = scatterPlot.generate(marker=dict(
                color=self.data[z].unique(), 
                cmin= self.data[z].min(),
                cmax= self.data[z].max(),
                colorscale="Plasma_r", showscale=False, symbol='square', size=10, opacity=0.4) , **other_kwargs)

    def create_x_axis_plot(self, x, z, class_kwargs) -> "figure":
        x_fig = super().create_x_axis_plot(x, z, class_kwargs)
        x_fig.update_xaxes(visible=False)
        
        return x_fig
    
    def create_y_axis_plot(self, y, z, class_kwargs) -> "figure":
        y_fig =  super().create_y_axis_plot(y, z, class_kwargs)
        y_fig.update_xaxes(range=[0, self.data[z].max()])
        y_fig.update_yaxes(range=[self.data[y].min(), self.data[y].max()])
        y_fig.update_layout(xaxis_title = self.ylabel, yaxis_title = self.zlabel)

        return y_fig

    def combine_plots(self, x_fig, y_fig):
        #############
        ##  Combine Plots
        
            # Create a figure with subplots
        fig_m = make_subplots(
            rows=2, cols=2,
            shared_xaxes=True, shared_yaxes=True,
            vertical_spacing=0, horizontal_spacing=0,
            subplot_titles=(None, f"Integrated {self.xlabel}", f"Integrated {self.ylabel}", None),
            specs=[[{}, {"type": "xy", "rowspan": 1, "secondary_y":True}],
                [{"type": "xy", "rowspan": 1, "secondary_y":False},     {"type": "xy", "rowspan": 1, "secondary_y":False}]]
        )
        
        # Add the heatmap to the first row
        for trace in self.fig.data:
            trace.showlegend = False
            trace.legendgroup = trace.name
            fig_m.add_trace(trace, row=2, col=2, secondary_y=False)
            
        # Update the heatmao layout
        fig_m.update_layout(self.fig.layout)
        fig_m.update_yaxes(row=2, col=2, secondary_y=False) 
        
        # Add the x-axis plot to the second row
        for trace in x_fig.data:

            trace.legendgroup = trace.name
            fig_m.add_trace(trace, row=1, col=2, secondary_y=True)
        
        # Update the XIC layout
        fig_m.update_layout(x_fig.layout)

        # Make the y-axis of fig_xic independent
        fig_m.update_yaxes(overwrite=True, row=1, col=2, secondary_y=True)
        
        # Manually adjust the domain of secondary y-axis to only span the first row of the subplot
        fig_m['layout']['yaxis3']['domain'] = [0.5, 1.0]
        
        # Add the XIM plot to the second row
        for trace in y_fig.data:
            trace.showlegend = False
            trace.legendgroup = trace.name
            fig_m.add_trace(trace, row=2, col=1)

        # Update the XIM layout
        fig_m.update_layout(y_fig.layout)

        # Make the x-axis of fig_xim independent
        fig_m.update_xaxes(overwrite=True, row=2, col=1)

        # Reverse the x-axis range for the XIM subplot
        fig_m.update_xaxes(autorange="reversed", row=2, col=1)

        # Update xaxis properties
        fig_m.update_xaxes(title_text=self.xlabel, row=2, col=2)
        fig_m.update_xaxes(title_text=self.zlabel,  row=2, col=1)

        # Update yaxis properties
        fig_m.update_yaxes(title_text=self.zlabel, row=1, col=2)
        fig_m.update_yaxes(title_text=self.ylabel, row=2, col=1)

        # Update the layout
        fig_m.update_layout(
            height=self.height,
            width=self.width,
            title=self.title
        )
        
        # Overwrite the figure with the new grid figure
        self.fig = fig_m
        
        self._update_plot_aes(self.fig)