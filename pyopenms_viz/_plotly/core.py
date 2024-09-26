from __future__ import annotations

from abc import ABC

from typing import List, Tuple, Union

import plotly.graph_objects as go
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

from pandas.core.frame import DataFrame

from numpy import column_stack, log, nan

from .._core import (
    BasePlot,
    LinePlot,
    VLinePlot,
    ScatterPlot,
    BaseMSPlot,
    ChromatogramPlot,
    MobilogramPlot,
    SpectrumPlot,
    PeakMapPlot,
    APPEND_PLOT_DOC,
)

from .._config import bokeh_line_dash_mapper
from .._misc import ColorGenerator, MarkerShapeGenerator, is_latex_formatted
from ..constants import PEAK_BOUNDARY_ICON, FEATURE_BOUNDARY_ICON


class PLOTLYPlot(BasePlot, ABC):
    """
    Base class for assembling a Ploty plot
    """

    @property
    def _interactive(self) -> bool:
        return True

    def _load_extension(self):
        """
        Tries to load the plotly extensions, if not throw an import error
        """
        try:
            import plotly.graph_objects
        except ImportError:
            raise ImportError(
                f"plotly is not installed. Please install using `pip install plotly` to use this plotting library in pyopenms-viz"
            )

    def _create_figure(self):
        """
        Create a new figure, if a figure is not supplied
        """
        fig = go.Figure()
        fig.update_layout(
            title=self.title,
            xaxis_title=self.xlabel,
            yaxis_title=self.ylabel,
            width=self.width,
            height=self.height,
            template="simple_white",
            dragmode="select",
        )
        return fig

    def _update_plot_aes(self, fig, **kwargs) -> None:
        """
        Update the plot aesthetics.
        """
        fig.update_layout(
            legend_title=self.legend_config.title,
            legend_font_size=self.legend_config.fontsize,
            showlegend=self.legend_config.show,
        )
        # Update to look similar to Bokeh theme
        # Customize the layout
        fig.update_layout(
            plot_bgcolor="#FFFFFF",  # Set the plot background color
            font_family="Helvetica",  # Set the font family
            # font_size=12,  # Set the font size
            title_font_family="Helvetica",  # Set the title font family
            title_font_size=self.title_font_size,  # Set the title font size
            xaxis_title_font_family="Helvetica",  # Set the x-axis title font family
            xaxis_title_font_size=self.xaxis_label_font_size,  # Set the x-axis title font size
            yaxis_title_font_family="Helvetica",  # Set the y-axis title font family
            yaxis_title_font_size=self.yaxis_label_font_size,  # Set the y-axis title font size
            xaxis_gridcolor="#CCCCCC",  # Set the x-axis grid color
            yaxis_gridcolor="#CCCCCC",  # Set the y-axis grid color
            xaxis_tickfont_size=self.xaxis_tick_font_size,  # Set the x-axis tick font size
            xaxis_tickfont_family="Helvetica",  # Set the x-axis tick font family
            yaxis_tickfont_size=self.yaxis_tick_font_size,  # Set the y-axis tick font size
            yaxis_tickfont_family="Helvetica",  # Set the y-axis tick font family
            legend_font_family="Helvetica",  # Set the legend font family
        )

        # Add x-axis grid lines and ticks
        fig.update_xaxes(
            showgrid=self.grid,  # Add x-axis grid lines
            showline=True,
            linewidth=1,
            linecolor="black",
            ticks="outside",  # Add x-axis ticks outside the plot area
            tickwidth=1,  # Set the width of x-axis ticks
            tickcolor="black",  # Set the color of x-axis ticks
        )

        # Add y-axis grid lines and ticks
        fig.update_yaxes(
            showgrid=self.grid,  # Add y-axis grid lines
            showline=True,
            linewidth=1,
            linecolor="black",
            tickwidth=1,  # Set the width of y-axis ticks
            tickcolor="black",  # Set the color of y-axis ticks
        )

    def _add_legend(self, fig, legend):
        pass

    def _add_tooltips(self, fig, tooltips, custom_hover_data=None):
        # In case figure is constructed of multiple traces (e.g. one trace per MS peak) add annotation for each point in trace
        if len(fig.data) > 1:
            for i in range(len(fig.data)):
                fig.data[i].update(
                    hovertemplate=tooltips,
                    customdata=[custom_hover_data[i, :]] * len(fig.data[i].x),
                )
            return
        fig.update_traces(hovertemplate=tooltips, customdata=custom_hover_data)

    def _add_bounding_box_drawer(self, fig, **kwargs):
        fig.update_layout(
            modebar_add=["drawrect", "eraseshape"],
            newshape=dict(
                showlegend=True,
                label=dict(
                    texttemplate="x0: %{x0:.2f} | x1: %{x1:.2f}<br>y0: %{y0:.2f} | y1: %{y1:.2f}",
                    textposition="top left",
                ),
                line_color="#F02D1A",
                fillcolor=None,
                line=dict(
                    dash="dash",
                ),
                opacity=0.5,
            ),
        )

    def _add_bounding_vertical_drawer(self, fig):
        # Note: self.label_suffix must be defined
        self.label_suffix = self.x  ### NOTE: not sure if this is correct behavior

        fig.add_trace(go.Scatter(x=[], y=[], mode="lines"))
        fig.update_layout(
            modebar_add=["drawrect", "eraseshape"],
            newshape=dict(
                showlegend=True,
                label=dict(
                    texttemplate=self.label_suffix
                    + "_0: %{x0:.2f} | "
                    + self.label_suffix
                    + "_1: %{x1:.2f}",
                    textposition="top left",
                ),
                line_color="#F02D1A",
                fillcolor=None,
                line=dict(
                    dash="dash",
                ),
                drawdirection="vertical",
                opacity=0.5,
            ),
        )

    def _modify_x_range(
        self,
        fig,
        x_range: Tuple[float, float] | None = None,
        padding: Tuple[float, float] | None = None,
    ):
        start, end = x_range
        if padding is not None:
            start = start - (start * padding[0])
            end = end + (end * padding[1])
        fig.update_xaxes(range=[start, end])

    def _modify_y_range(
        self,
        fig,
        y_range: Tuple[float, float] | None = None,
        padding: Tuple[float, float] | None = None,
    ):
        start, end = y_range
        if padding is not None:
            start = start - (start * padding[0])
            end = end + (end * padding[1])
        fig.update_yaxes(range=[start, end])

    def show_default(self, fig):
        fig.show()

    def show_sphinx(self):
        return self.fig


class PLOTLYLinePlot(PLOTLYPlot, LinePlot):
    """
    Class for assembling a set of line plots in plotly
    """

    @APPEND_PLOT_DOC
    def plot(  # type: ignore[override]
        self,
        fig,
    ) -> Tuple[Figure, "Legend"]:  # note legend is always none for consistency

        traces = []
        if self.by is None:
            trace = go.Scatter(
                x=self.data[self.x],
                y=self.data[self.y],
                mode="lines",
                line=dict(color=self.current_color),
            )
            traces.append(trace)
        else:
            for group, df in self.data.groupby(self.by):
                trace = go.Scatter(
                    x=df[self.x],
                    y=df[self.y],
                    mode="lines",
                    name=group,
                    line=dict(color=self.current_color),
                )
                traces.append(trace)

        fig.add_traces(data=traces)
        return fig, None


class PLOTLYVLinePlot(PLOTLYPlot, VLinePlot):

    @APPEND_PLOT_DOC
    def plot(self, fig) -> Tuple[Figure, "Legend"]:

        if not self.plot_3d:
            traces = []
            if self.by is None:
                for _, row in self.data.iterrows():
                    if self.direction == "horizontal":
                        x_data = [0, row[self.x]]
                        y_data = [row[self.y]] * 2
                    else:
                        x_data = [row[self.x]] * 2
                        y_data = [0, row[self.y]]

                    trace = go.Scattergl(
                        x=x_data,
                        y=y_data,
                        mode="lines",
                        name="",
                        showlegend=False,
                        line=dict(color=self.current_color),
                    )
                    traces.append(trace)
            else:
                show_legend = self.legend_config.show
                for group, df in self.data.groupby(self.by):
                    for _, row in df.iterrows():
                        if self.direction == "horizontal":
                            x_data = [0, row[self.x]]
                            y_data = [row[self.y]] * 2
                        else:
                            x_data = [row[self.x]] * 2
                            y_data = [0, row[self.y]]

                        trace = go.Scattergl(
                            x=x_data,
                            y=y_data,
                            mode="lines",
                            name=group,
                            legendgroup=group,
                            showlegend=show_legend,
                            line=dict(color=self.current_color),
                        )
                        show_legend = False  # only show the legend for one trace
                        traces.append(trace)

            fig.add_traces(data=traces)
        else:
            if self.by is None:
                x_vert = []
                y_vert = []
                z_vert = []
                z_min = self.data[self.z].min()
                z_max = self.data[self.z].max()
                for x_val, y_val, z_val in zip(
                    self.data[self.x], self.data[self.y], self.data[self.z]
                ):
                    for i in range(2):
                        x_vert.append(x_val)
                        y_vert.append(y_val)
                        if i == 0:
                            z_vert.append(0)
                        else:
                            z_vert.append(z_val)
                    x_vert.append(None)
                    y_vert.append(None)
                    z_vert.append(None)

                fig.add_trace(
                    go.Scatter3d(
                        x=x_vert,
                        y=y_vert,
                        z=z_vert,
                        mode="lines",
                        line=dict(
                            width=5,
                            color=[z if z is not None else 0 for z in z_vert],
                            colorscale="magma_r",
                            cmin=z_min,
                            cmax=z_max,
                        ),
                        name="",
                        showlegend=False,
                    )
                )
            else:
                for group, df in self.data.groupby(self.by):
                    # Transform to vertical line data with no connections
                    x_vert = []
                    y_vert = []
                    z_vert = []
                    for (
                        x_val,
                        y_val,
                        z_val,
                    ) in zip(df[self.x], df[self.y], df[self.z]):
                        for i in range(2):
                            x_vert.append(x_val)
                            y_vert.append(y_val)
                            if i == 0:
                                z_vert.append(0)
                            else:
                                z_vert.append(z_val)
                        x_vert.append(None)
                        y_vert.append(None)
                        z_vert.append(None)

                    fig.add_trace(
                        go.Scatter3d(
                            x=x_vert,
                            y=y_vert,
                            z=z_vert,
                            mode="lines",
                            line=dict(width=5, color=self.current_color),
                            name=group,
                            legendgroup=group,
                        )
                    )

            # Add gridlines
            fig.update_layout(
                scene=dict(
                    xaxis=dict(
                        title=self.xlabel,
                        nticks=4,
                        gridcolor="rgb(255, 255, 255)",
                        zerolinecolor="rgb(255, 255, 255)",
                    ),
                    yaxis=dict(
                        title=self.ylabel,
                        nticks=4,
                        gridcolor="rgb(255, 255, 255)",
                        zerolinecolor="rgb(255, 255, 255)",
                    ),
                    zaxis=dict(
                        title=self.zlabel,
                        nticks=4,
                        gridcolor="rgb(255, 255, 255)",
                        zerolinecolor="rgb(255, 255, 255)",
                    ),
                )
            )

            camera = dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.25, y=1.8, z=1.25),
            )
            fig.update_layout(scene_camera=camera)

        return fig, None

    def _add_annotations(
        self,
        fig,
        ann_texts: list[str],
        ann_xs: list[float],
        ann_ys: list[float],
        ann_colors: list[str],
    ):
        annotations = []
        for text, x, y, color in zip(ann_texts, ann_xs, ann_ys, ann_colors):
            if text is not nan and text != "" and text != "nan":
                # Check if the text contains LaTeX-style expressions
                if is_latex_formatted(text):
                    # Wrap the text in '$' to indicate LaTeX math mode
                    text = r"${}$".format(text)
                annotation = go.layout.Annotation(
                    text=text.replace("\n", "<br>"),
                    x=x,
                    y=y,
                    showarrow=False,
                    xanchor="left",
                    font=dict(
                        family="Open Sans Mono, monospace",
                        size=self.annotation_font_size,
                        color=color,
                    ),
                )
                annotations.append(annotation)

        for annotation in annotations:
            fig.add_annotation(annotation)
        return fig


class PLOTLYScatterPlot(PLOTLYPlot, ScatterPlot):

    def __post_init__(self):
        super().__post_init__()
        if self.marker is None:
            self.marker = MarkerShapeGenerator(engine="PLOTLY")

    @APPEND_PLOT_DOC
    def plot(self, fig) -> Tuple[Figure, "Legend"]:

        marker_dict = dict()
        # Check for z-dimension and plot heatmap
        # Plotting heatmaps with z dimension overwrites marker_dict.
        if self.z:
            # Default values for heatmap
            heatmap_defaults = dict(
                color=self.data[self.z],
                colorscale="Inferno_r",
                showscale=False,
                size=self.marker_size,
                opacity=0.8,
                cmin=self.data[self.z].min(),
                cmax=self.data[self.z].max(),
            )
            # If no marker_dict was in kwargs, use default for heatmaps
            if self.marker is None:
                marker_dict = heatmap_defaults
            # Else update existing marker dict with default values if key is missing
            else:
                for k, v in heatmap_defaults.items():
                    if k not in marker_dict.keys():
                        marker_dict[k] = v

        marker_dict = {}
        marker_dict["color"] = self.data[self.z] if self.z else self.current_color
        traces = []
        if self.by is None:
            print(self.marker)
            marker_dict["symbol"] = self.current_marker
            trace = go.Scattergl(
                x=self.data[self.x],
                y=self.data[self.y],
                mode="markers",
                marker=marker_dict,
                showlegend=False,
            )
            traces.append(trace)
        else:
            for group, df in self.data.groupby(self.by):
                if self.z is not None:
                    marker_dict["color"] = self.df[self.z]
                trace = go.Scatter(
                    x=df[self.x],
                    y=df[self.y],
                    mode="markers",
                    name=group,
                    marker=dict(symbol=self.current_marker, color=self.current_color),
                )
                traces.append(trace)

        fig.add_traces(data=traces)
        return fig, None


class PLOTLY_MSPlot(BaseMSPlot, PLOTLYPlot, ABC):

    def get_line_renderer(self, **kwargs) -> None:
        return PLOTLYLinePlot(**kwargs)

    def get_vline_renderer(self, **kwargs) -> None:
        return PLOTLYVLinePlot(**kwargs)

    def get_scatter_renderer(self, **kwargs) -> None:
        return PLOTLYScatterPlot(**kwargs)

    def plot_x_axis_line(self, fig):
        fig.add_hline(y=0, line_color="black", line=dict(width=1))

    def _create_tooltips(self, entries, index=True):
        custom_hover_data = []
        # Add data from index if required
        if index:
            custom_hover_data.append(self.data.index)
        # Get the rest of the columns
        custom_hover_data += [self.data[col] for col in entries.values()]

        tooltips = []
        # Add tooltip text for index if required
        if index:
            tooltips.append("index: %{customdata[0]}")

        custom_hover_data_index = 1 if index else 0

        for key in entries.keys():
            tooltips.append(
                f"{key}" + ": %{customdata[" + str(custom_hover_data_index) + "]}"
            )
            custom_hover_data_index += 1

        return "<br>".join(tooltips), column_stack(custom_hover_data)


class PLOTLYChromatogramPlot(PLOTLY_MSPlot, ChromatogramPlot):

    def _add_peak_boundaries(self, fig, annotation_data):
        color_gen = ColorGenerator(
            colormap=self.annotation_colormap, n=annotation_data.shape[0]
        )
        for idx, (_, feature) in enumerate(annotation_data.iterrows()):
            if "q_value" in annotation_data.columns:
                legend_label = f"Feature {idx} (q-value: {feature['q_value']:.4f})"
            else:
                legend_label = f"Feature {idx}"
            fig.add_trace(
                go.Scatter(
                    mode="lines",
                    x=[
                        feature["leftWidth"],
                        feature["leftWidth"],
                        feature["rightWidth"],
                        feature["rightWidth"],
                    ],
                    y=[feature["apexIntensity"], 0, 0, feature["apexIntensity"]],
                    opacity=0.5,
                    line=dict(
                        color=next(color_gen),
                        dash=bokeh_line_dash_mapper(
                            self.plot_config.annotation_line_type, "plotly"
                        ),
                        width=self.plot_config.annotation_line_width,
                    ),
                    name=legend_label,
                )
            )

    def get_manual_bounding_box_coords(self, arg):
        # TODO: Implement this method, plotly doesn't have a direct easy way of extracting the relayout events. Would need to implement / add a dash dependency to add a callback to extract the relayout events
        pass


class PLOTLYMobilogramPlot(PLOTLYChromatogramPlot, MobilogramPlot):
    pass


class PLOTLYSpectrumPlot(PLOTLY_MSPlot, SpectrumPlot):
    def _prepare_data(
        self, spectrum: DataFrame, x: str, y: str, reference_spectrum: DataFrame | None
    ) -> Tuple[List]:
        spectrum, reference_spectrum = super()._prepare_data(
            spectrum, x, y, reference_spectrum
        )

        if reference_spectrum is not None:
            # add a "ref" label to legend elements, useful when for plotly because reference elements and base elements are in the same legend
            if self.by is not None:
                reference_spectrum[self.by] = reference_spectrum[self.by] + " (ref)"

        return spectrum, reference_spectrum


class PLOTLYPeakMapPlot(PLOTLY_MSPlot, PeakMapPlot):

    def create_main_plot(self):
        if not self.plot_3d:
            scatterPlot = self.get_scatter_renderer(
                data=self.data, x=self.x, y=self.y, z=self.z
            )
            fig = scatterPlot.generate(None, None)

            if self.z is not None:
                tooltips, custom_hover_data = self._create_tooltips(
                    {self.xlabel: self.x, self.ylabel: self.y, self.zlabel: self.z}
                )
            else:
                tooltips, custom_hover_data = self._create_tooltips(
                    {self.xlabel: self.x, self.ylabel: self.y}
                )

            self._add_tooltips(fig, tooltips, custom_hover_data=custom_hover_data)

            if self.annotation_data is not None:
                self._add_box_boundaries(fig, self.annotation_data)
            return fig
        else:
            vlinePlot = self.get_vline_renderer(data=self.data, x=self.x, y=self.y)
            return vlinePlot.generate(None, None)

            # TODO: Custom tooltips currently not working as expected for 3D plot, it has it's own tooltip that works out of the box, but with set x, y, z name to value
            # tooltips, custom_hover_data = self._create_tooltips({self.xlabel: x, self.ylabel: y, self.zlabel: z})
            # self._add_tooltips(fig, tooltips, custom_hover_data=custom_hover_data

    def create_x_axis_plot(self) -> "figure":
        x_fig = super().create_x_axis_plot()
        x_fig.update_xaxes(visible=False)

        return x_fig

    def create_y_axis_plot(self, main_fig) -> "figure":
        y_fig = super().create_y_axis_plot(main_fig)
        y_fig.update_xaxes(range=[0, self.data[self.z].max()])
        y_fig.update_yaxes(range=[self.data[self.y].min(), self.data[self.y].max()])
        y_fig.update_layout(
            xaxis_title=self.y_plot_config.xlabel, yaxis_title=self.y_plot_config.ylabel
        )

        return y_fig

    def combine_plots(self, main_fig, x_fig, y_fig):
        #############
        ##  Combine Plots

        # Create a figure with subplots
        fig_m = make_subplots(
            rows=2,
            cols=2,
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0,
            horizontal_spacing=0,
            subplot_titles=(
                None,
                f"Integrated {self.xlabel}",
                f"Integrated {self.ylabel}",
                None,
            ),
            specs=[
                [{}, {"type": "xy", "rowspan": 1, "secondary_y": True}],
                [
                    {"type": "xy", "rowspan": 1, "secondary_y": False},
                    {"type": "xy", "rowspan": 1, "secondary_y": False},
                ],
            ],
        )

        # Add the heatmap to the first row
        for trace in main_fig.data:
            trace.showlegend = False
            trace.legendgroup = trace.name
            fig_m.add_trace(trace, row=2, col=2, secondary_y=False)

        # Update the heatmao layout
        fig_m.update_layout(main_fig.layout)
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
        fig_m["layout"]["yaxis3"]["domain"] = [0.5, 1.0]

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
        fig_m.update_xaxes(title_text=self.zlabel, row=2, col=1)

        # Update yaxis properties
        fig_m.update_yaxes(title_text=self.zlabel, row=1, col=2)
        fig_m.update_yaxes(title_text=self.ylabel, row=2, col=1)

        # Remove axes for first quadrant
        fig_m.update_xaxes(visible=False, row=1, col=1)
        fig_m.update_yaxes(visible=False, row=1, col=1)

        # Update the layout
        fig_m.update_layout(height=self.height, width=self.width, title=self.title)

        # change subplot axes font size
        # each plot title is treated as an annotation
        fig_m.for_each_annotation(
            lambda annotation: annotation.update(font=dict(size=self.title_font_size))
        )
        fig_m.for_each_xaxis(
            lambda axis: axis.title.update(font=dict(size=self.xaxis_label_font_size))
        )
        fig_m.for_each_yaxis(
            lambda axis: axis.title.update(font=dict(size=self.yaxis_label_font_size))
        )
        fig_m.for_each_xaxis(
            lambda axis: axis.tickfont.update(size=self.xaxis_tick_font_size)
        )
        fig_m.for_each_yaxis(
            lambda axis: axis.tickfont.update(size=self.yaxis_tick_font_size)
        )

        self._update_plot_aes(fig_m)
        return fig_m

    def _add_box_boundaries(self, fig, annotation_data):
        color_gen = ColorGenerator(
            colormap=self.feature_config.colormap, n=annotation_data.shape[0]
        )
        for idx, (_, feature) in enumerate(annotation_data.iterrows()):
            x0 = feature["leftWidth"]
            x1 = feature["rightWidth"]
            y0 = feature["IM_leftWidth"]
            y1 = feature["IM_rightWidth"]

            color = next(color_gen)

            if "name" in annotation_data.columns:
                use_name = feature["name"]
            else:
                use_name = f"Feature {idx}"
            if "q_value" in annotation_data.columns:
                legend_label = f"{use_name} (q-value: {feature['q_value']:.4f})"
            else:
                legend_label = f"{use_name}"
            fig.add_trace(
                go.Scatter(
                    x=[
                        x0,
                        x1,
                        x1,
                        x0,
                        x0,
                    ],  # Start and end at the same point to close the shape
                    y=[y0, y0, y1, y1, y0],
                    mode="lines",
                    fill="none",
                    opacity=0.5,
                    line=dict(
                        color=color,
                        width=self.feature_config.line_width,
                        dash=bokeh_line_dash_mapper(
                            self.feature_config.line_type, "plotly"
                        ),
                    ),
                    name=legend_label,
                )
            )
