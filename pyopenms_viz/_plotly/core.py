from __future__ import annotations

from abc import ABC
from typing import Tuple

import plotly.graph_objects as go
from numpy import column_stack, nan
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

from .._config import bokeh_line_dash_mapper
from .._core import (
    APPEND_PLOT_DOC,
    BaseMSPlot,
    BasePlot,
    ChromatogramPlot,
    LinePlot,
    MobilogramPlot,
    PeakMapPlot,
    ScatterPlot,
    SpectrumPlot,
    VLinePlot,
)
from .._misc import ColorGenerator, MarkerShapeGenerator, is_latex_formatted


class PLOTLYPlot(BasePlot, ABC):
    """
    Base class for assembling a Ploty plot
    """

    # In plotly the canvas is referred to as a figure
    @property
    def fig(self):
        return self.canvas

    @fig.setter
    def fig(self, value):
        self.canvas = value
        self._config.canvas = value

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
                "plotly is not installed. Please install using `pip install plotly` to use this plotting library in pyopenms-viz"
            )

    def _create_figure(self):
        """
        Create a new figure, if a figure is not supplied
        """
        self.fig = go.Figure()
        self.fig.update_layout(
            title=self.title,
            xaxis_title=self.xlabel,
            yaxis_title=self.ylabel,
            width=self.width,
            height=self.height,
            template="simple_white",
            dragmode="select",
        )

    def _update_plot_aes(self, **kwargs) -> None:
        """
        Update the plot aesthetics.
        """
        self.fig.update_layout(
            legend_title=self.legend_config.title,
            legend_font_size=self.legend_config.fontsize,
            showlegend=self.legend_config.show,
        )
        # Update to look similar to Bokeh theme
        # Customize the layout
        self.fig.update_layout(
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
        self.fig.update_xaxes(
            showgrid=self.grid,  # Add x-axis grid lines
            showline=True,
            linewidth=1,
            linecolor="black",
            ticks="outside",  # Add x-axis ticks outside the plot area
            tickwidth=1,  # Set the width of x-axis ticks
            tickcolor="black",  # Set the color of x-axis ticks
        )

        # Add y-axis grid lines and ticks
        self.fig.update_yaxes(
            showgrid=self.grid,  # Add y-axis grid lines
            showline=True,
            linewidth=1,
            linecolor="black",
            tickwidth=1,  # Set the width of y-axis ticks
            tickcolor="black",  # Set the color of y-axis ticks
        )

    def generate(
        self, tooltips, custom_hover_data, fixed_tooltip_for_trace=True
    ) -> Figure:
        """
        Generate the Plotly plot with optional interactive tooltips.

        Args:
            tooltips: A Plotly hovertemplate string that defines the tooltip format.
                Can reference customdata fields using %{customdata[0]}, etc.
            custom_hover_data: A numpy array of additional data for hover tooltips.
                Shape depends on fixed_tooltip_for_trace setting.
            fixed_tooltip_for_trace (bool): If True, each trace gets one row of
                custom_hover_data repeated for all points. If False, custom_hover_data
                rows are distributed sequentially across all trace points.

        Returns:
            Figure: The generated Plotly figure.
        """
        self._load_extension()
        if self.canvas is None:
            self._create_figure()

        self.plot()
        self._update_plot_aes()

        if tooltips is not None and self._interactive:
            self._add_tooltips(tooltips, custom_hover_data, fixed_tooltip_for_trace)
        return self.canvas

    def _add_legend(self, legend):
        pass

    def _add_tooltips(
        self, tooltips, custom_hover_data=None, fixed_tooltip_for_trace=True
    ):
        # In case figure is constructed of multiple traces (e.g. one trace per MS peak) add annotation for each point in trace
        if len(self.fig.data) > 1:
            if fixed_tooltip_for_trace:
                for i in range(len(self.fig.data)):
                    self.fig.data[i].update(
                        hovertemplate=tooltips,
                        customdata=[custom_hover_data[i, :]] * len(self.fig.data[i].x),
                    )
                return
            else:
                counter = 0
                for i in range(len(self.fig.data)):
                    l = len(self.fig.data[i].x)
                    self.fig.data[i].update(
                        hovertemplate=tooltips,
                        customdata=custom_hover_data[counter : counter + l, :],
                    )
                    counter += l
                return
        self.fig.update_traces(hovertemplate=tooltips, customdata=custom_hover_data)

    def _add_bounding_box_drawer(self, **kwargs):
        self.fig.update_layout(
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

    def _add_bounding_vertical_drawer(self):
        # Note: self.label_suffix must be defined
        self.label_suffix = self.x  ### NOTE: not sure if this is correct behavior

        self.fig.add_trace(go.Scatter(x=[], y=[], mode="lines"))
        self.fig.update_layout(
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
        x_range: Tuple[float, float] | None = None,
        padding: Tuple[float, float] | None = None,
    ):
        start, end = x_range
        if padding is not None:
            start = start - (start * padding[0])
            end = end + (end * padding[1])
        self.fig.update_xaxes(range=[start, end])

    def _modify_y_range(
        self,
        y_range: Tuple[float, float] | None = None,
        padding: Tuple[float, float] | None = None,
    ):
        start, end = y_range
        if padding is not None:
            start = start - (start * padding[0])
            end = end + (end * padding[1])
        self.fig.update_yaxes(range=[start, end])

    def show_default(self):
        self.fig.show()

    def show_sphinx(self):
        return self.fig

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
                if is_latex_formatted(text):
                    # NOTE: Plotly uses MathJax for LaTeX rendering. Newlines are rendered as \\.
                    text = text.replace("\n", r" \\\ ")
                    text = r"${}$".format(text)
                else:
                    text = text.replace("\n", "<br>")
                annotation = go.layout.Annotation(
                    text=text,
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


class PLOTLYLinePlot(PLOTLYPlot, LinePlot):
    """
    Class for assembling a set of line plots in plotly
    """

    @APPEND_PLOT_DOC
    def plot(self):
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
            for group, df in self.data.groupby(self.by, sort=False):
                trace = go.Scatter(
                    x=df[self.x],
                    y=df[self.y],
                    mode="lines",
                    name=group,
                    line=dict(color=self.current_color),
                )
                traces.append(trace)

        self.fig.add_traces(data=traces)


class PLOTLYVLinePlot(PLOTLYPlot, VLinePlot):
    @APPEND_PLOT_DOC
    def plot(self):
        if not self.plot_3d:
            traces = []
            use_color = self.current_color
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
                        line=dict(color=use_color),
                    )
                    traces.append(trace)
            else:
                show_legend = self.legend_config.show
                for group, df in self.data.groupby(self.by):
                    use_color = self.current_color
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
                            line=dict(color=use_color),
                        )
                        show_legend = False  # only show the legend for one trace
                        traces.append(trace)

            self.fig.add_traces(data=traces)
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

                self.fig.add_trace(
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

                    self.fig.add_trace(
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
            self.fig.update_layout(
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
            self.fig.update_layout(scene_camera=camera)


class PLOTLYScatterPlot(PLOTLYPlot, ScatterPlot):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        if (
            isinstance(self.marker, MarkerShapeGenerator)
            and not self.marker.is_initialized()
        ):
            self.marker.initialize_shape_cycle_from_engine("PLOTLY")

    @APPEND_PLOT_DOC
    def plot(self):
        marker_dict = dict()
        # Check for z-dimension and plot heatmap
        # Plotting heatmaps with z dimension overwrites marker_dict.
        if self.z:
            # Default values for heatmap
            heatmap_defaults = dict(
                color=self.data[self.z],
                colorscale="Inferno_r",
                showscale=False,
                size=self.marker_size
                / 3,  # divide by 3 so approximately same unit as other backends
                opacity=self.opacity,
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

        traces = []
        if self.by is None:
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
                    marker_dict["color"] = (
                        df[self.z] if self.z is not None else self.current_color
                    )
                    marker_dict["symbol"] = self.current_marker
                trace = go.Scatter(
                    x=df[self.x],
                    y=df[self.y],
                    mode="markers",
                    name=group,
                    marker=marker_dict,
                )
                traces.append(trace)

        self.fig.add_traces(data=traces)


class PLOTLY_MSPlot(BaseMSPlot, PLOTLYPlot, ABC):
    def get_line_renderer(self, **kwargs) -> None:
        return PLOTLYLinePlot(**kwargs)

    def get_vline_renderer(self, **kwargs) -> None:
        return PLOTLYVLinePlot(**kwargs)

    def get_scatter_renderer(self, **kwargs) -> None:
        return PLOTLYScatterPlot(**kwargs)

    def plot_x_axis_line(self, fig, line_color="#EEEEEE", line_width=1.5, opacity=1):
        fig.add_hline(
            y=0, line_color=line_color, line=dict(width=line_width), opacity=opacity
        )

    def _create_tooltips(self, entries, index=True, data=None):
        # Use provided data or fall back to self.data
        if data is None:
            data = self.data
        
        custom_hover_data = []
        # Add data from index if required
        if index:
            custom_hover_data.append(data.index)
        # Get the rest of the columns
        custom_hover_data += [data[col] for col in entries.values()]

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
    def _add_peak_boundaries(self, annotation_data):
        super()._add_peak_boundaries(annotation_data)
        color_gen = ColorGenerator(
            colormap=self.annotation_colormap, n=annotation_data.shape[0]
        )
        for idx, (_, feature) in enumerate(annotation_data.iterrows()):
            if "q_value" in annotation_data.columns:
                legend_label = f"Feature {idx} (q-value: {feature['q_value']:.4f})"
            else:
                legend_label = f"Feature {idx}"
            self.fig.add_trace(
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
                            self.annotation_line_type, "plotly"
                        ),
                        width=self.annotation_line_width,
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
    def _prepare_data(self, df, label_suffix=" (ref)"):
        df = super()._prepare_data(df, label_suffix)
        if self.reference_spectrum is not None and self.by is not None:
            self.reference_spectrum[self.by] = (
                self.reference_spectrum[self.by] + label_suffix
            )
        return df


class PLOTLYPeakMapPlot(PLOTLY_MSPlot, PeakMapPlot):
    # NOTE: canvas is only used in matplotlib backend
    def create_main_plot(self, canvas=None) -> Figure:
        if not self.plot_3d:
            scatterPlot = self.get_scatter_renderer(data=self.data, config=self._config)
            self.fig = scatterPlot.generate(None, None)

            if self.z is not None:
                tooltips, custom_hover_data = self._create_tooltips(
                    {self.xlabel: self.x, self.ylabel: self.y, self.zlabel: self.z}
                )
            else:
                tooltips, custom_hover_data = self._create_tooltips(
                    {self.xlabel: self.x, self.ylabel: self.y}
                )

            self._add_tooltips(tooltips, custom_hover_data=custom_hover_data)

            if self.annotation_data is not None:
                self._add_box_boundaries(self.annotation_data)
        else:
            vlinePlot = self.get_vline_renderer(
                data=self.data, x=self.x, y=self.y, config=self._config
            )
            self.fig = vlinePlot.generate(None, None)
            if self.annotation_data is not None:
                a_x, a_y, a_z, a_t, a_c = self._compute_3D_annotations(
                    self.annotation_data, self.x, self.y, self.z
                )
                vlinePlot._add_annotations(self.fig, a_t, a_x, a_y, a_c, a_z)

        return self.fig

        # TODO: Custom tooltips currently not working as expected for 3D plot, it has it's own tooltip that works out of the box, but with set x, y, z name to value
        # tooltips, custom_hover_data = self._create_tooltips({self.xlabel: x, self.ylabel: y, self.zlabel: z})
        # self._add_tooltips(fig, tooltips, custom_hover_data=custom_hover_data

    def create_x_axis_plot(self) -> Figure:
        x_fig = super().create_x_axis_plot()
        x_fig.update_xaxes(visible=False)

        return x_fig

    def create_y_axis_plot(self) -> Figure:
        y_fig = super().create_y_axis_plot()
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

        self.fig = fig_m
        self._update_plot_aes()
        return fig_m

    def _add_box_boundaries(self, annotation_data):
        color_gen = ColorGenerator(
            colormap=self.annotation_colormap, n=annotation_data.shape[0]
        )
        for idx, (_, feature) in enumerate(annotation_data.iterrows()):
            x0 = feature[self.annotation_x_lb]
            x1 = feature[self.annotation_x_ub]
            y0 = feature[self.annotation_y_lb]
            y1 = feature[self.annotation_y_ub]

            if self.annotation_colors in feature:
                color = feature[self.annotation_colors]
            else:
                color = next(color_gen)

            if self.annotation_names in annotation_data.columns:
                use_name = feature[self.annotation_names]
            else:
                use_name = f"Feature {idx}"
            if "q_value" in annotation_data.columns:
                legend_label = f"{use_name} (q-value: {feature['q_value']:.4f})"
            else:
                legend_label = f"{use_name}"
            # Plot rectangle
            self.fig.add_shape(
                type="rect",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                line=dict(
                    color=color,
                    width=self.annotation_line_width,
                    dash=bokeh_line_dash_mapper(self.annotation_line_type, "plotly"),
                ),
                fillcolor="rgba(0,0,0,0)",
                opacity=0.5,
                layer="above",
            )
            # Add a dummy trace for the legend
            self.fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(
                        color=color,
                        width=self.annotation_line_width,
                        dash=bokeh_line_dash_mapper(
                            self.annotation_line_type, "plotly"
                        ),
                    ),
                    showlegend=True,
                    name=legend_label,
                )
            )
