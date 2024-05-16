import re
from dataclasses import dataclass
from itertools import cycle
from typing import List, Literal, Optional, Union

import pandas as pd
import plotly.graph_objects as go

from .BasePlotter import Colors, _BasePlotter, _BasePlotterConfig
import matplotlib.pyplot as plt
from bokeh.plotting import figure
from bokeh.models import Span, Label, Range1d
import streamlit as st
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
        super().__init__(config=config, **kwargs)

    def _get_ion_color_annotation(self, annotation: str) -> str:
        """
        Retrieve the color associated with a specific ion annotation from a predefined colormap.

        This function maps a given ion annotation type to its corresponding color. The mapping is based on
        conventional color assignments used in mass spectrometry to differentiate between various ion types.
        If the provided annotation does not exactly match any predefined type, the function attempts to match
        based on the first character of the annotation. If no matches are found, a default color is returned.

        Parameters:
        - annotation (str): The ion annotation for which the color needs to be retrieved. This could be a
        single character representing the type of ion (e.g., 'a', 'b', 'x', etc.), or a more complex
        string from which the first character will be extracted to determine the ion type.

        Returns:
        - str: A hexadecimal color code as a string. This color is associated with the type of ion in the
        provided annotation. If the annotation cannot be matched, a default color ("#555555") is returned.

        Raises:
        - KeyError: If the first character of a complex annotation string does not match any predefined keys
        and is not handled by the function. However, the function is designed to handle unmatchable cases by
        returning a default color, so this exception would typically not be raised.

        Example usage:
        >>> _get_ion_color_annotation("y")
        '#EF553B'  # Assuming '#EF553B' is the color assigned to 'y' type ions
        """
        colormap = {
            "a": Colors["PURPLE"],
            "b": Colors["BLUE"],
            "c": Colors["LIGHTBLUE"],
            "x": Colors["YELLOW"],
            "y": Colors["RED"],
            "z": Colors["ORANGE"],
            "none": "black",
            "matched": Colors["DARKGRAY"],
            "unmatched": Colors["DARKGRAY"],
        }
        for key in colormap.keys():
            # Exact matches
            if annotation == key:
                return colormap[key]
            # Fragment ions via regex
            x = re.search(r"^[abcxyz]{1}[0-9]*[+-]$", annotation)
            if x:
                return colormap[annotation[0]]
        return Colors["DARKGRAY"]

    def _get_annotation_text(self, peak: pd.Series) -> str:
        if "custom_annotation" in peak.index and self.config.custom_annotation_text:
            text = peak["custom_annotation"]
        else:
            text = ""
            if self.config.annotate_ions:
                if peak["ion_annotation"] != "none":
                    text = f"{peak['ion_annotation']}"
            if self.config.annotate_sequence:
                if peak["sequence"]:
                    if text:
                        text += "<br>"
                    text += peak["sequence"]
            if self.config.annotate_mz:
                if text:
                    text += "<br>"
                text += str(peak["mz"])
        return text


    def _get_annotation_color(self, peak: pd.Series, fallback_color: str = "black") -> str:
        if "color_annotation" in peak.index and self.config.custom_annotation_color:
            color = peak["color_annotation"]
        elif self.config.annotate_ions:
            color = self._get_ion_color_annotation(peak["ion_annotation"])
        else:
            return fallback_color if not self.config.custom_peak_color else peak["color_peak"]
        return color

    def _get_relative_intensity_ticks(self) -> tuple[List[int], List[str]]:
        ticks = [0, 25, 50, 75, 100]
        labels = ["0%", "25%", "50%", "75%", "100%"]
        if self.config.mirror_spectrum:
            ticks = [-100, -75, -50, -25] + ticks
            labels = labels[-4:][::-1] + labels
        return ticks, labels

    def _plotMatplotlib(
        self,
        spectrum: Union[pd.DataFrame, List[pd.DataFrame]],
        reference_spectrum: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
    ):
        if not isinstance(spectrum, list):
            spectrum = [spectrum]
        if (not isinstance(reference_spectrum, list)) and reference_spectrum is not None:
            reference_spectrum = [reference_spectrum]
        elif reference_spectrum is None:
            reference_spectrum = []

        if self.config.relative_intensity or self.config.mirror_spectrum:
            for df in spectrum + reference_spectrum:
                df["intensity"] = df["intensity"] / df["intensity"].max() * 100

        fig, ax = plt.subplots(
            figsize=(self.config.width / 100, self.config.height / 100)
        )

        def plot_spectrum(ax, df, color, mirror=False):
            for i, peak in df.iterrows():
                intensity = -peak["intensity"] if mirror else peak["intensity"]
                peak_color = (
                    color
                    if not (
                        self.config.custom_peak_color and "color_peak" in peak.index
                    )
                    else peak["color_peak"]
                )
                ax.plot(
                    [peak["mz"], peak["mz"]],
                    [0, intensity],
                    color=peak_color,
                    linewidth=1.5,
                    label=peak["native_id"] if i == 0 else None,
                )
                if (
                    self.config.annotate_mz
                    or self.config.annotate_ions
                    or self.config.annotate_sequence
                    or self.config.custom_annotation_text
                ):
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
        gs_colors = self._get_n_grayscale_colors(max([len(spectrum), len(reference_spectrum)]))
        colors = cycle(gs_colors)
        for spec in spectrum:
            plot_spectrum(ax, spec, next(colors))

        if self.config.mirror_spectrum:
            colors = cycle(gs_colors)
            for ref_spec in reference_spectrum:
                plot_spectrum(ax, ref_spec, next(colors), mirror=True)
            # Plot zero line
            plt.plot(ax.get_xlim(), [0, 0], color="#EEEEEE", linewidth=1.5)
        
        # Format y-axis
        if self.config.relative_intensity or self.config.mirror_spectrum:
            ticks, labels = self._get_relative_intensity_ticks()
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)
        else:
            ax.ticklabel_format(axis="both", style="sci", useMathText=True)

        ax.set_title(self.config.title, fontsize=12, loc="left", pad=20)
        ax.set_xlabel(self.config.xlabel, fontsize=10, style="italic", color=Colors["DARKGRAY"])
        ax.set_ylabel(self.config.ylabel, fontsize=10, color=Colors["DARKGRAY"])
        ax.xaxis.label.set_color(Colors["DARKGRAY"])
        ax.tick_params(axis="x", colors=Colors["DARKGRAY"])
        ax.yaxis.label.set_color(Colors["DARKGRAY"])
        ax.tick_params(axis="y", colors=Colors["DARKGRAY"])
        ax.set_ylim([0 if not self.config.mirror_spectrum else None, None])
        ax.spines[["right", "top"]].set_visible(False)
        ax.legend(loc="best") if self.config.show_legend else None

        return fig

    def _plotBokeh(
        self,
        spectrum: Union[pd.DataFrame, List[pd.DataFrame]],
        reference_spectrum: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
    ):
        if not isinstance(spectrum, list):
            spectrum = [spectrum]
        if (not isinstance(reference_spectrum, list)) and reference_spectrum is not None:
            reference_spectrum = [reference_spectrum]
        elif reference_spectrum is None:
            reference_spectrum = []

        if self.config.relative_intensity or self.config.mirror_spectrum:
            for df in spectrum + reference_spectrum:
                df["intensity"] = df["intensity"] / df["intensity"].max() * 100
        # Initialize figure
        p = figure(
            title=self.config.title,
            x_axis_label=self.config.xlabel,
            y_axis_label=self.config.ylabel,
            width=self.config.width,
            height=self.config.height,
        )

        def plot_spectrum(p, df, color, mirror=False):
            for i, peak in df.iterrows():
                intensity = -peak["intensity"] if mirror else peak["intensity"]
                peak_color = (
                    color
                    if not (self.config.custom_peak_color and "color_peak" in peak.index)
                    else peak["color_peak"]
                )
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
                if (
                    self.config.annotate_mz
                    or self.config.annotate_ions
                    or self.config.annotate_sequence
                    or self.config.custom_annotation_text
                ):
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

        # Plot spectra with annotations
        gs_colors = self._get_n_grayscale_colors(max([len(spectrum), len(reference_spectrum)]))
        colors = cycle(gs_colors)
        for spec in spectrum:
            plot_spectrum(p, spec, next(colors))
        # Plot mirror spectra with annotations
        if self.config.mirror_spectrum:
            colors = cycle(gs_colors)
            for ref_spec in reference_spectrum:
                plot_spectrum(p, ref_spec, next(colors), mirror=True)
            # Plot zero line
            zero_line = Span(location=0, dimension='width', line_color='#EEEEEE', line_width=1.5)
            p.add_layout(zero_line)

        # Format y-axis
        if self.config.relative_intensity or self.config.mirror_spectrum:
            ticks, labels = self._get_relative_intensity_ticks()
            p.yaxis.ticker = ticks
            p.yaxis.major_label_overrides = {tick: label for tick, label in zip(ticks, labels)}
        else:
            p.yaxis.formatter.use_scientific = True
        
        # Set y-axis limits
        if self.config.mirror_spectrum:
            p.y_range.start = -110
        else:
            p.y_range.start = 0
            
        p.grid.grid_line_color = None

        # Show legend if configured
        if self.config.show_legend:
            p.legend.location = "top_right"
        else:
            p.legend.visible = False

        return p

    def _plotPlotly(
        self,
        spectrum: Union[pd.DataFrame, List[pd.DataFrame]],
        reference_spectrum: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
    ):
        def _create_peak_traces(
            spectrum: pd.DataFrame,
            line_color: str,
            intensity_direction: Literal[1, -1] = 1,
        ) -> List[go.Scatter]:
            return [
                go.Scatter(
                    x=[peak["mz"]] * 2,
                    y=[0, intensity_direction * peak["intensity"]],
                    mode="lines",
                    line=dict(
                        color=(
                            peak["color_peak"]
                            if "color_peak" in peak.index
                            and self.config.custom_peak_color
                            else line_color
                        )
                    ),
                    name=peak["native_id"],
                    text=f"m/z: {peak['mz']}<br>intensity: {peak['intensity']}<br>{peak['native_id']}",
                    hoverinfo="text",
                    showlegend=True if i == 0 else False,
                )
                for i, peak in spectrum.iterrows()
            ]

        def _create_annotations(
            spectra: List[pd.DataFrame],
            intensity_sign: Literal[1, -1] = 1,
        ) -> List[dict]:
            if (
                not self.config.annotate_mz
                and not self.config.annotate_ions
                and not self.config.annotate_sequence
                and not self.config.custom_annotation_text
            ):
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

        # Make sure both spectrum and reference spectrum are lists containing DataFrames.
        if not isinstance(spectrum, list):
            spectrum = [spectrum]
        if (not isinstance(reference_spectrum, list)) and reference_spectrum is not None:
            reference_spectrum = [reference_spectrum]
        elif reference_spectrum is None:
            reference_spectrum = []

        # If relative intensity is set, convert intensity values for all DataFrames.
        if self.config.relative_intensity or self.config.mirror_spectrum:
            for df in spectrum + reference_spectrum:
                df["intensity"] = df["intensity"].apply(
                    lambda x: x / max(df["intensity"]) * 100
                )

        # Create figure with initial layout
        layout = go.Layout(
            title=dict(text=self.config.title),
            xaxis=dict(
                title=self.config.xlabel,
                autorangeoptions=dict(
                    include=[
                        min([min(df["mz"]) for df in spectrum + reference_spectrum])
                        - 1,
                        max([max(df["mz"]) for df in spectrum + reference_spectrum])
                        + 1,
                    ]
                ),
                constrain="domain",  # Optional, ensures that the range cannot be changed by user zoom
                title_standoff=5,
            ),
            yaxis=dict(
                title=self.config.ylabel,
                constrain="domain",  # Optional, ensures that the range cannot be changed by user zoom
                title_standoff=10,
            ),
            showlegend=self.config.show_legend,
            template="simple_white",
        )
        fig = go.Figure(layout=layout)
        # Get grayscale colors for maximum number of spectra on one axis
        self.gs_colors = self._get_n_grayscale_colors(max([len(spectrum), len(reference_spectrum)]))
        # Peak traces
        traces = []
        # Spectra
        colors = cycle(self.gs_colors)
        for spec in spectrum:
            color = next(colors)
            traces += _create_peak_traces(spec, color)
        # Mirror spectra
        if self.config.mirror_spectrum:
            colors = cycle(self.gs_colors)
            for spec in reference_spectrum:
                color = next(colors)
                traces += _create_peak_traces(spec, color, intensity_direction=-1)
        fig.add_traces(traces)
        # Annotations
        for annotation in _create_annotations(spectrum, 1):
            fig.add_annotation(annotation)
        if self.config.mirror_spectrum:
            for annotation in _create_annotations(reference_spectrum, -1):
                fig.add_annotation(annotation)
            # Draw horizontal line
            fig.add_hline(y=0, line_color=Colors["LIGHTGRAY"], line_width=2)
        # Format y-axis
        if self.config.relative_intensity or self.config.mirror_spectrum:
            ticks, labels = self._get_relative_intensity_ticks()
            fig.update_layout(yaxis=dict(tickmode="array", tickvals=ticks, ticktext=labels), yaxis_range=[-110 if self.config.mirror_spectrum else 0,110])
        else:
            fig.update_layout(yaxis=dict(range=[0, None]))

        return fig


# ============================================================================= #
## FUNCTIONAL API ##
# ============================================================================= #


def plotSpectrum(
    spectrum: Union[pd.DataFrame, List[pd.DataFrame]],
    reference_spectrum: Union[pd.DataFrame, List[pd.DataFrame]] = None,
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
        spectrum (Union[pd.DataFrame, List[pd.DataFrame]]): OpenMS MSSpectrum Object
        reference_spectrum (Union[pd.DataFrame, List[pd.DataFrame]], optional): Optional OpenMS Spectrum object to plot in mirror or used in annotation. Defaults to None.
        ion_mobility (bool, optional): If true, plots a heatmap of m/z vs ion mobility with intensity as color. Defaults to False.
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
        ylabel (str, optional): Y-axis label. Defaults to "intensity".
        show_legend (int, optional): Show legend. Defaults to False.
        engine (Literal['PLOTLY', 'BOKEH'], optional): Plotting engine to use. Defaults to 'PLOTLY' can be either 'PLOTLY' or 'BOKEH'

    Returns:
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
