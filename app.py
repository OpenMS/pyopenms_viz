import streamlit as st
import pandas as pd
import numpy as np
from pyopenms_viz.MSExperimentPlotter import plotMSExperiment
from pyopenms_viz.SpectrumPlotter import plotSpectrum
import time
import pyopenms as oms
from urllib.request import urlretrieve

st.title("pyopenms-viz demo")


def load_demo_mzML():
    with st.spinner("Loading example MSExperiment data..."):
        gh = "https://raw.githubusercontent.com/OpenMS/pyopenms-docs/master"
        mzML_path = gh + "/src/data/FeatureFinderMetaboIdent_1_input.mzML"
        urlretrieve(mzML_path, "ms_data.mzML")

        exp = oms.MSExperiment()
        oms.MzMLFile().load("ms_data.mzML", exp)

        st.session_state.exp_df = exp.get_df(long=True)
    # exp.to_parquet("peakmap.parquet")


def display_fig(fig, engine):
    if engine == "MATPLOTLIB":
        st.pyplot(fig, use_container_width=True)
    elif engine == "BOKEH":
        st.bokeh_chart(fig, use_container_width=True)
    else:
        st.plotly_chart(fig, use_container_width=True)


if "exp_df" not in st.session_state:
    load_demo_mzML()

demo = st.selectbox("select demo", ["MSExperiment", "MSSpectrum"])

if demo == "MSExperiment":

    @st.experimental_fragment()
    def msexpdemo():
        cols = st.columns(3)
        show_legend = cols[0].checkbox("show_legend", True)
        relative_intensity = cols[1].checkbox("relative_intensity", False)
        plot3D = cols[2].checkbox("plot3D", False)
        bin_peaks = cols[0].selectbox("bin_peaks", ["auto", "True", "False"])
        num_RT_bins = cols[1].number_input("num_RT_bins", 10, 100, 50, 10)
        num_mz_bins = cols[2].number_input("num_mz_bins", 10, 100, 50, 10)
        engine = cols[0].selectbox("engine", ["MATPLOTLIB", "BOKEH", "PLOTLY"])
        # if cols[2].button("Update Plot", type="primary", use_container_width=True):
        if bin_peaks == "True":
            bin_peaks = True
        elif bin_peaks == "False":
            bin_peaks = False
        fig = plotMSExperiment(
            st.session_state.exp_df,
            engine=engine,
            show_legend=show_legend,
            relative_intensity=relative_intensity,
            plot3D=plot3D,
            num_RT_bins=num_RT_bins,
            num_mz_bins=num_mz_bins,
            bin_peaks=bin_peaks,
            height=800
        )
        if engine != "PLOTLY":
            display_fig(fig, engine)
            return
        if "msexp_selection" in st.session_state:
            points = st.session_state.msexp_selection.selection.points
            box = st.session_state.msexp_selection.selection.box
            if box:
                df = st.session_state.exp_df.copy()
                df = df[df["RT"] > box[0]["x"][0]]
                df = df[df["mz"] > box[0]["y"][1]]
                df = df[df["mz"] < box[0]["y"][0]]
                df = df[df["RT"] < box[0]["x"][1]]

                fig = plotMSExperiment(
                    df,
                    engine=engine,
                    show_legend=show_legend,
                    relative_intensity=relative_intensity,
                    plot3D=plot3D,
                    num_RT_bins=num_RT_bins,
                    num_mz_bins=num_mz_bins,
                    bin_peaks=bin_peaks,
                    height=800
                )
        st.plotly_chart(
            fig,
            use_container_width=True,
            key="msexp_selection",
            selection_mode=["points", "box"],
            on_select="rerun",
            config={
                "displaylogo": False,
                "modeBarButtonsToRemove": [
                    "zoom",
                    "pan",
                    # "select",
                    "lasso",
                    "zoomin",
                    "autoscale",
                    "zoomout",
                    "resetscale",
                ],
            },
        )

    msexpdemo()
    # st.write(st.session_state.msexp_selection.points)
    with st.expander("`exp.get_df(long=True)`", expanded=False):
        st.dataframe(st.session_state.exp_df)
    with st.expander("`plotMSExperiment()`", expanded=False):
        st.write(plotMSExperiment)

elif demo == "MSSpectrum":
    spec = pd.DataFrame(
        {
            "mz": [50.989, 74.1324, 100.5332, 101.545, 102.5343, 200.4232],
            "intensity": [10, 20, 25, 12, 6, 17],
            "ion_mobility": [2, 4, 17, 1, 3, 1],
            "ion_annotation": ["a+", "b3+", "c5+", "y9+", "z3+", "x4+"],
        }
    )
    spec["ion_mobility_unit"] = "ms"
    spec["precursor_mz"] = 221.08
    spec["precursor_charge"] = 1
    spec["native_id"] = "spec_0"
    spec["spectrum"] = "ABC"
    spec["color_peak"] = "green"
    spec["custom_annotation"] = "custom"
    spec["color_annotation"] = "blue"
    spec["sequence"] = "DMAGCH"

    @st.experimental_fragment
    def msspecdemo():
        cols = st.columns(3)
        show_legend = cols[0].checkbox("show_legend", True)
        relative_intensity = cols[1].checkbox("relative_intensity", False)
        ion_mobility = cols[2].checkbox("ion_mobility")
        annotate_sequence = cols[0].checkbox("annotate_sequence")
        mirror_spectrum = cols[1].checkbox("mirror_spectrum")
        custom_annotation_text = cols[2].checkbox("custom_annotation_text")
        custom_peak_color = cols[0].checkbox("custom_peak_color")
        annotate_ions = cols[1].checkbox("annotate_ions")
        custom_annotation_color = cols[2].checkbox("custom_annotation_color")
        annotate_mz = cols[0].checkbox("annotate_mz")
        engine = cols[0].selectbox("engine", ["MATPLOTLIB", "BOKEH", "PLOTLY"])
        if cols[2].button("Update Plot", type="primary", use_container_width=True):
            fig = plotSpectrum(
                spec,
                spec,
                engine=engine,
                relative_intensity=relative_intensity,
                show_legend=show_legend,
                ion_mobility=ion_mobility,
                annotate_sequence=annotate_sequence,
                mirror_spectrum=mirror_spectrum,
                custom_annotation_text=custom_annotation_text,
                custom_peak_color=custom_peak_color,
                annotate_ions=annotate_ions,
                custom_annotation_color=custom_annotation_color,
                annotate_mz=annotate_mz,
            )
            display_fig(fig, engine)

    msspecdemo()

    with st.expander("MSSpectrum df", expanded=False):
        st.dataframe(spec)
    with st.expander("`plotSpectrum()`", expanded=False):
        st.write(plotSpectrum)
