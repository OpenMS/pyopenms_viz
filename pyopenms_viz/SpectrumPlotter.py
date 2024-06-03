import re
from dataclasses import dataclass
from itertools import cycle
from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from bokeh.models import ColorBar, ColumnDataSource, HoverTool, Label, Span
from bokeh.palettes import Plasma256
from bokeh.plotting import figure
from bokeh.transform import linear_cmap

from .BasePlotter import Colors, _BasePlotter, _BasePlotterConfig


@dataclass(kw_only=True)
class SpectrumPlotterConfig(_BasePlotterConfig):
    ion_mobility: bool = False
    annotate_mz: bool = False
    annotate_ions: bool = False
    annotate_sequence: bool = False
    mirror_spectrum: bool = (False,)
    custom_peak_color: bool = (False,)
    custom_annotation_color: bool = (False,)
    custom_annotation_text: bool = False


class SpectrumPlotter(_BasePlotter):
    def __init__(self, config: SpectrumPlotterConfig, **kwargs) -> None:
        """
        Initialize the SpectrumPlotter with a given configuration and optional parameters.

        Args:
            config (SpectrumPlotterConfig): Configuration settings for the spectrum plotter.
            **kwargs: Additional keyword arguments for customization.
        """
        super().__init__(config=config, **kwargs)
        # If y-axis label is default ("intensity") and ion_mobility is True, update label
        if self.config.ylabel == "intensity" and self.config.ion_mobility:
            self.config.ylabel = "ion mobility"

    def _get_ion_color_annotation(self, annotation: str) -> str:
        """
        Retrieve the color associated with a specific ion annotation from a predefined colormap.

        Args:
            annotation (str): The ion annotation for which the color needs to be retrieved.

        Returns:
            str: A hexadecimal color code associated with the ion type.
        """
        colormap = {
            "a": Colors["PURPLE"],
            "b": Colors["BLUE"],
            "c": Colors["LIGHTBLUE"],
            "x": Colors["YELLOW"],
            "y": Colors["RED"],
            "z": Colors["ORANGE"],
        }
        return colormap.get(annotation.lower(), Colors["DARKGRAY"])

    def _get_peak_color(self, default_color: str, peak: pd.Series) -> str:
        """
        Determine the color of a peak based on custom settings or annotation.

        Args:
            default_color (str): The default color for peaks.
            peak (pd.Series): The peak data.

        Returns:
            str: The color of the peak.
        """
        if self.config.custom_peak_color and "color_peak" in peak:
            return peak["color_peak"]
        
        if self.config.annotate_ions:
            return self._get_annotation_color(peak, default_color)
        
        return default_color

    def _get_annotation_text(self, peak: pd.Series) -> str:
        """
        Generate the annotation text for a given peak based on the configuration.

        Args:
            peak (pd.Series): The peak data.

        Returns:
            str: The annotation text.
        """
        if "custom_annotation" in peak and self.config.custom_annotation_text:
            return peak["custom_annotation"]
        
        texts = []
        
        if self.config.annotate_ions and peak["ion_annotation"] != "none":
            texts.append(peak["ion_annotation"])
        
        if self.config.annotate_sequence and peak["sequence"]:
            texts.append(peak["sequence"])
        
        if self.config.annotate_mz:
            texts.append(str(peak["mz"]))
        
        return "<br>".join(texts)

    def _get_annotation_color(
        self, peak: pd.Series, fallback_color: str = "black"
    ) -> str:
        """
        Determine the color for annotations based on custom settings or ion type.

        Args:
            peak (pd.Series): The peak data.
            fallback_color (str, optional): The fallback color if no custom color is set. Defaults to "black".

        Returns:
            str: The color of the annotation.
        """
        if "color_annotation" in peak and self.config.custom_annotation_color:
            return peak["color_annotation"]

        if self.config.annotate_ions:
            return self._get_ion_color_annotation(peak["ion_annotation"])

        if self.config.custom_peak_color and "color_peak" in peak:
            return peak["color_peak"]

        return fallback_color

    def _get_relative_intensity_ticks(self) -> tuple[list[int], list[str]]:
        """
        Generate the ticks and labels for relative intensity on the y-axis.

        Returns:
            tuple[list[int], list[str]]: The ticks and corresponding labels.
        """
        ticks = [0, 25, 50, 75, 100]
        labels = ["0%", "25%", "50%", "75%", "100%"]

        if self.config.mirror_spectrum:
            mirror_ticks = [-100, -75, -50, -25]
            mirror_labels = ["-100%", "-75%", "-50%", "-25%"]
            ticks = mirror_ticks + ticks
            labels = mirror_labels + labels

        return ticks, labels

    def _ensure_list_format(
        self,
        spectrum: Union[pd.DataFrame, list[pd.DataFrame]],
        reference_spectrum: Union[pd.DataFrame, list[pd.DataFrame], None],
    ) -> tuple[list, list]:
        """
        Ensure that the input spectra and reference spectra are in list format.

        Args:
            spectrum (Union[pd.DataFrame, list[pd.DataFrame]]): The main spectrum data.
            reference_spectrum (Union[pd.DataFrame, list[pd.DataFrame], None]): The reference spectrum data.

        Returns:
            tuple[list, list]: The spectra and reference spectra in list format.
        """
        if not isinstance(spectrum, list):
            spectrum = [spectrum]
        
        if reference_spectrum is None:
            reference_spectrum = []
        elif not isinstance(reference_spectrum, list):
            reference_spectrum = [reference_spectrum]
        
        return spectrum, reference_spectrum

    def _check_relative_intensity(
        self,
        spectrum: Union[pd.DataFrame, list[pd.DataFrame]],
        reference_spectrum: Union[pd.DataFrame, list[pd.DataFrame], None],
    ) -> tuple[list, list]:
        """
        Convert intensities to relative intensities if necessary.

        Args:
            spectrum (Union[pd.DataFrame, list[pd.DataFrame]]): The main spectrum data.
            reference_spectrum (Union[pd.DataFrame, list[pd.DataFrame], None]): The reference spectrum data.

        Returns:
            tuple[list, list]: The spectra and reference spectra with relative intensities if configured.
        """
        if self.config.relative_intensity or self.config.mirror_spectrum:
            combined_spectra = spectrum + (reference_spectrum if reference_spectrum else [])
            for df in combined_spectra:
                df["intensity"] = df["intensity"] / df["intensity"].max() * 100
                
        return spectrum, reference_spectrum

    def _combine_sort_spectra_by_intensity(
        self, spectra: list[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Combine and sort spectra by intensity.

        Args:
            spectra (list[pd.DataFrame]): list of spectra dataframes.

        Returns:
            pd.DataFrame: Combined and sorted spectrum dataframe.
        """
        combined_df = pd.concat(spectra).reset_index(drop=True)
        sorted_df = combined_df.sort_values(by="intensity").reset_index(drop=True)
        return sorted_df

    def _plotMatplotlib(
        self,
        spectrum: Union[pd.DataFrame, list[pd.DataFrame]],
        reference_spectrum: Optional[Union[pd.DataFrame, list[pd.DataFrame]]] = None,
    ) -> plt.Figure:
        """
        Plot the spectrum using Matplotlib.

        Args:
            spectrum (Union[pd.DataFrame, list[pd.DataFrame]]): The main spectrum data.
            reference_spectrum (Optional[Union[pd.DataFrame, list[pd.DataFrame]]]): The reference spectrum data. Defaults to None.

        Returns:
            plt.Figure: The Matplotlib figure object.
        """

        def plot_spectrum(ax, df, color, mirror=False):
            for i, peak in df.iterrows():
                intensity = -peak["intensity"] if mirror else peak["intensity"]
                peak_color = self._get_peak_color(color, peak)
                ax.plot(
                    [peak["mz"], peak["mz"]],
                    [0, intensity],
                    color=peak_color,
                    linewidth=1.5,
                    label=peak["native_id"] if i == 0 else None,
                )
                if any([self.config.annotate_mz, self.config.annotate_ions, self.config.annotate_sequence, self.config.custom_annotation_text]):
                    text = self._get_annotation_text(peak).replace("<br>", "\n")
                    annotation_color = self._get_annotation_color(peak, peak_color)
                    ax.annotate(
                        text,
                        xy=(peak["mz"], intensity),
                        xytext=(1, 0),
                        textcoords="offset points",
                        fontsize=8,
                        color=annotation_color,
                    )

        spectrum, reference_spectrum = self._ensure_list_format(spectrum, reference_spectrum)
        spectrum, reference_spectrum = self._check_relative_intensity(spectrum, reference_spectrum)

        fig, ax = plt.subplots(figsize=(self.config.width / 100, self.config.height / 100))
        ax.set_title(self.config.title, fontsize=12, loc="left", pad=20)
        ax.set_xlabel(self.config.xlabel, fontsize=10, style="italic", color=Colors["DARKGRAY"])
        ax.set_ylabel(self.config.ylabel, fontsize=10, color=Colors["DARKGRAY"])
        ax.xaxis.label.set_color(Colors["DARKGRAY"])
        ax.tick_params(axis="x", colors=Colors["DARKGRAY"])
        ax.yaxis.label.set_color(Colors["DARKGRAY"])
        ax.tick_params(axis="y", colors=Colors["DARKGRAY"])
        ax.spines[["right", "top"]].set_visible(False)

        if self.config.ion_mobility:
            df = self._combine_sort_spectra_by_intensity(spectrum)
            scatter = ax.scatter(
                df["mz"],
                df["ion_mobility"],
                c=df["intensity"],
                cmap="plasma_r",
                s=20,
                marker="s",
            )
            if self.config.show_legend:
                cb = fig.colorbar(scatter, aspect=40)
                cb.outline.set_visible(False)
            return fig

        gs_colors = self._get_n_grayscale_colors(max(len(spectrum), len(reference_spectrum or [])))
        colors = cycle(gs_colors)

        for spec in spectrum:
            plot_spectrum(ax, spec, next(colors))

        if self.config.mirror_spectrum:
            colors = cycle(gs_colors)
            for ref_spec in reference_spectrum:
                plot_spectrum(ax, ref_spec, next(colors), mirror=True)
            ax.plot(ax.get_xlim(), [0, 0], color="#EEEEEE", linewidth=1.5)

        ax.set_ylim([0 if not self.config.mirror_spectrum else None, None])
        if self.config.relative_intensity or self.config.mirror_spectrum:
            ticks, labels = self._get_relative_intensity_ticks()
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)
        else:
            ax.ticklabel_format(axis="both", style="sci", useMathText=True)

        if self.config.show_legend:
            ax.legend(loc="best")

        return fig

    def _plotBokeh(
        self,
        spectrum: Union[pd.DataFrame, list[pd.DataFrame]],
        reference_spectrum: Optional[Union[pd.DataFrame, list[pd.DataFrame]]] = None,
    ) -> figure:
        """
        Plot the spectrum using Bokeh.

        Args:
            spectrum (Union[pd.DataFrame, list[pd.DataFrame]]): The main spectrum data.
            reference_spectrum (Optional[Union[pd.DataFrame, list[pd.DataFrame]]]): The reference spectrum data. Defaults to None.

        Returns:
            bokeh.plotting.figure: The Bokeh figure object.
        """

        def plot_spectrum(p, df, color, mirror=False):
            for i, peak in df.iterrows():
                intensity = -peak["intensity"] if mirror else peak["intensity"]
                peak_color = self._get_peak_color(color, peak)
                if i == 0:
                    p.line(
                        [peak["mz"], peak["mz"]],
                        [0, intensity],
                        line_color=peak_color,
                        line_width=2,
                        legend_label=peak["native_id"],
                    )
                else:
                    p.line(
                        [peak["mz"], peak["mz"]],
                        [0, intensity],
                        line_color=peak_color,
                        line_width=2,
                    )
                if any([self.config.annotate_mz, self.config.annotate_ions, self.config.annotate_sequence, self.config.custom_annotation_text]):
                    text = self._get_annotation_text(peak).replace("<br>", "\n")
                    annotation_color = self._get_annotation_color(peak, peak_color)
                    label = Label(
                        x=peak["mz"],
                        y=intensity,
                        text=text,
                        text_font_size="8pt",
                        text_color=annotation_color,
                        x_offset=1,
                        y_offset=0,
                    )
                    p.add_layout(label)

        spectrum, reference_spectrum = self._ensure_list_format(spectrum, reference_spectrum)
        spectrum, reference_spectrum = self._check_relative_intensity(spectrum, reference_spectrum)

        # Initialize figure
        p = figure(
            title=self.config.title,
            x_axis_label=self.config.xlabel,
            y_axis_label=self.config.ylabel,
            width=self.config.width,
            height=self.config.height,
        )
        
        p.grid.grid_line_color = None
        p.border_fill_color = None
        p.outline_line_color = None

        if self.config.show_legend:
            p.legend.location = "top_right"
        else:
            p.legend.visible = False

        if self.config.ion_mobility:
            df = self._combine_sort_spectra_by_intensity(spectrum)
            mapper = linear_cmap(
                field_name="intensity",
                palette=Plasma256[::-1],
                low=df["intensity"].min(),
                high=df["intensity"].max(),
            )
            df["hover_text"] = df.apply(
                lambda x: f"{x['native_id']}<br>m/z: {x['mz']}<br>ion mobility: {x['ion_mobility']}<br>intensity: {x['intensity']}",
                axis=1,
            )
            source = ColumnDataSource(df)
            scatter = p.scatter(
                x="mz",
                y="ion_mobility",
                size=6,
                source=source,
                color=mapper,
                marker="square",
            )
            hover = HoverTool(
                tooltips="""
                <div>
                    <span>@hover_text{safe}</span>
                </div>
                """
            )
            p.add_tools(hover)
            if self.config.show_legend:
                color_bar = ColorBar(
                    color_mapper=mapper["transform"], width=8, location=(0, 0)
                )
                p.add_layout(color_bar, "right")
            return p

        gs_colors = self._get_n_grayscale_colors(max(len(spectrum), len(reference_spectrum or [])))
        colors = cycle(gs_colors)

        for spec in spectrum:
            plot_spectrum(p, spec, next(colors))

        if self.config.mirror_spectrum:
            colors = cycle(gs_colors)
            for ref_spec in reference_spectrum:
                plot_spectrum(p, ref_spec, next(colors), mirror=True)
            zero_line = Span(
                location=0, dimension="width", line_color="#EEEEEE", line_width=1.5
            )
            p.add_layout(zero_line)

        if self.config.relative_intensity or self.config.mirror_spectrum:
            ticks, labels = self._get_relative_intensity_ticks()
            p.yaxis.ticker = ticks
            p.yaxis.major_label_overrides = {
                tick: label for tick, label in zip(ticks, labels)
            }
        else:
            p.yaxis.formatter.use_scientific = True

        p.y_range.start = -110 if self.config.mirror_spectrum else 0

        return p

    def _plotPlotly(
        self,
        spectrum: Union[pd.DataFrame, list[pd.DataFrame]],
        reference_spectrum: Optional[Union[pd.DataFrame, list[pd.DataFrame]]] = None,
    ) -> go.Figure:
        """
        Plot the spectrum using Plotly.

        Args:
            spectrum (Union[pd.DataFrame, list[pd.DataFrame]]): The main spectrum data.
            reference_spectrum (Optional[Union[pd.DataFrame, list[pd.DataFrame]]]): The reference spectrum data. Defaults to None.

        Returns:
            go.Figure: The Plotly figure object.
        """

        def _create_peak_traces(
            spectrum: pd.DataFrame,
            line_color: str,
            intensity_direction: Literal[1, -1] = 1,
        ) -> list[go.Scattergl]:
            return [
                go.Scattergl(
                    x=[peak["mz"]] * 2,
                    y=[0, intensity_direction * peak["intensity"]],
                    mode="lines",
                    line=dict(color=self._get_peak_color(line_color, peak)),
                    name=peak["native_id"],
                    text=f"{peak['native_id']}<br>m/z: {peak['mz']}<br>intensity: {peak['intensity']}",
                    hoverinfo="text",
                    showlegend=(i == 0),
                )
                for i, peak in spectrum.iterrows()
            ]

        def _create_annotations(
            spectra: list[pd.DataFrame],
            intensity_sign: Literal[1, -1] = 1,
        ) -> list[dict]:
            if not any([
                self.config.annotate_mz,
                self.config.annotate_ions,
                self.config.annotate_sequence,
                self.config.custom_annotation_text
            ]):
                return []
            
            annotations = []
            colors = cycle(self.gs_colors)
            for spectrum in spectra:
                for _, peak in spectrum.iterrows():
                    text = self._get_annotation_text(peak)
                    color = self._get_annotation_color(peak, next(colors))
                    annotations.append(
                        dict(
                            x=peak["mz"],
                            y=intensity_sign * peak["intensity"],
                            text=text,
                            showarrow=False,
                            xanchor="left",
                            font=dict(
                                family="Open Sans Mono, monospace",
                                size=12,
                                color=color,
                            ),
                        )
                    )
            return annotations

        spectrum, reference_spectrum = self._ensure_list_format(spectrum, reference_spectrum)
        spectrum, reference_spectrum = self._check_relative_intensity(spectrum, reference_spectrum)

        layout = go.Layout(
            title=dict(text=self.config.title),
            xaxis=dict(title=self.config.xlabel),
            yaxis=dict(title=self.config.ylabel),
            showlegend=self.config.show_legend,
            template="simple_white",
        )
        fig = go.Figure(layout=layout)

        if self.config.ion_mobility:
            df = self._combine_sort_spectra_by_intensity(spectrum)
            df["hover_text"] = df.apply(
                lambda x: f"{x['native_id']}<br>m/z: {x['mz']}<br>ion mobility: {x['ion_mobility']}<br>intensity: {x['intensity']}",
                axis=1,
            )
            fig.add_trace(
                go.Scattergl(
                    name="peaks",
                    x=df["mz"],
                    y=df["ion_mobility"],
                    mode="markers",
                    marker=dict(
                        color=df["intensity"],
                        colorscale="sunset",
                        size=8,
                        symbol="square",
                        colorbar=dict(thickness=8, outlinewidth=0) if self.config.show_legend else None,
                    ),
                    hovertext=df["hover_text"],
                    hoverinfo="text",
                    showlegend=False,
                )
            )
            return fig

        self.gs_colors = self._get_n_grayscale_colors(max(len(spectrum), len(reference_spectrum or [])))
        colors = cycle(self.gs_colors)
        
        traces = []
        for spec in spectrum:
            traces += _create_peak_traces(spec, next(colors))

        if self.config.mirror_spectrum:
            colors = cycle(self.gs_colors)
            for ref_spec in reference_spectrum:
                traces += _create_peak_traces(ref_spec, next(colors), intensity_direction=-1)

        fig.add_traces(traces)

        annotations = _create_annotations(spectrum, 1)
        for annotation in annotations:
            fig.add_annotation(annotation)

        if self.config.mirror_spectrum:
            annotations = _create_annotations(reference_spectrum, -1)
            for annotation in annotations:
                fig.add_annotation(annotation)
            fig.add_hline(y=0, line_color=Colors["LIGHTGRAY"], line_width=2)

        if self.config.relative_intensity or self.config.mirror_spectrum:
            ticks, labels = self._get_relative_intensity_ticks()
            fig.update_layout(
                yaxis=dict(tickmode="array", tickvals=ticks, ticktext=labels),
                yaxis_range=[-110 if self.config.mirror_spectrum else 0, 110],
            )
        else:
            fig.update_yaxes(rangemode="nonnegative")

        return fig



# ============================================================================= #
## FUNCTIONAL API ##
# ============================================================================= #


def plotSpectrum(
    spectrum: Union[pd.DataFrame, list[pd.DataFrame]],
    reference_spectrum: Union[pd.DataFrame, list[pd.DataFrame]] = None,
    ion_mobility: bool = False,
    annotate_mz: bool = False,
    annotate_ions: bool = False,
    annotate_sequence: bool = False,
    mirror_spectrum: bool = False,
    relative_intensity: bool = False,
    custom_peak_color: bool = False,
    custom_annotation_text: bool = False,
    custom_annotation_color: bool = False,
    width: int = 750,
    height: int = 500,
    title: str = "Spectrum Plot",
    xlabel: str = "m/z",
    ylabel: str = "intensity",
    show_legend: bool = False,
    engine: Literal["PLOTLY", "BOKEH"] = "PLOTLY",
):
    """
    Plots a Spectrum from an MSSpectrum object

    Args:
        spectrum (Union[pd.DataFrame, list[pd.DataFrame]]): OpenMS MSSpectrum Object
        reference_spectrum (Union[pd.DataFrame, list[pd.DataFrame]], optional): Optional OpenMS Spectrum object to plot in mirror or used in annotation. Defaults to None.
        ion_mobility (bool, optional): If true, plots spectra (not including reference spectra) as heatmap of m/z vs ion mobility with intensity as color. Defaults to False.
        annotate_mz (bool, optional): If true, annotate peaks with m/z values. Defaults to False.
        annotate_ions (bool, optional): If true, annotate fragment ions. Defaults to False.
        annotate_sequence (bool, optional): Annotate peaks based on sequence provided. Defaults to False
        mirror_spectrum (bool, optional): If true, plot mirror spectrum. Defaults to True, if no mirror reference_spectrum is provided, this is ignored.
        relative_intensity (bool, optional): If true, plot relative intensity values. Defaults to False.
        custom_peak_color (bool, optional): If true, plot peaks with colors from "color_peak" column.
        custom_annotation_text (bool, optional): If true, annotate peaks with custom text from "custom_annotation" column. Overwrites all other annotations.Use <br> for line breaks.
        custom_annotation_color (bool, optional): If true, plot annotations with colors from "color_annotation" column.
        width (int, optional): Width of plot. Defaults to 500px.
        height (int, optional): Height of plot. Defaults to 500px.
        title (str, optional): Plot title. Defaults to "Spectrum Plot".
        xlabel (str, optional): X-axis label. Defaults to "m/z".
        ylabel (str, optional): Y-axis label. Defaults to "intensity" or "ion mobility".
        show_legend (int, optional): Show legend. Defaults to False.
        engine (Literal['PLOTLY', 'BOKEH'], optional): Plotting engine to use. Defaults to 'PLOTLY' can be either 'PLOTLY' or 'BOKEH'

    Returns:
        Plot: The generated plot using the specified engine.
    """
    config = SpectrumPlotterConfig(
        ion_mobility=ion_mobility,
        annotate_mz=annotate_mz,
        mirror_spectrum=mirror_spectrum,
        annotate_sequence=annotate_sequence,
        annotate_ions=annotate_ions,
        relative_intensity=relative_intensity,
        custom_peak_color=custom_peak_color,
        custom_annotation_text=custom_annotation_text,
        custom_annotation_color=custom_annotation_color,
        width=width,
        height=height,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        show_legend=show_legend,
        engine=engine,
    )
    plotter = SpectrumPlotter(config)
    return plotter.plot(spectrum, reference_spectrum=reference_spectrum)
