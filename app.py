import streamlit as st
import pandas as pd
import numpy as np
from pyopenms_viz.MSExperimentPlotter import plotMSExperiment
from pyopenms_viz.SpectrumPlotter import plotSpectrum
import pyopenms as oms
from urllib.request import urlretrieve

with st.sidebar:
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
        st.pyplot(fig)
    elif engine == "BOKEH":
        st.bokeh_chart(fig)
    else:
        st.plotly_chart(fig)


def get_common_parameters():
    params = {}
    with st.sidebar:
        params["engine"] = st.selectbox("engine", ["MATPLOTLIB", "BOKEH", "PLOTLY"])
        params["width"] = st.number_input("width", 50, 1000, 500, 50)
        params["height"] = st.number_input("height", 50, 1000, 500, 50)
        params["title"] = st.text_input("title", "Title")
        params["xlabel"] = st.text_input("xlabel", "x-label")
        params["ylabel"] = st.text_input("xlabel", "y-label")
        params["show_legend"] = st.checkbox("show_legend", False)
    return params


def get_MSExperiment_params():
    params = {}
    params["plot3D"] = st.checkbox("plot3D", False)
    params["relative_intensity"] = st.checkbox("relative_intensity", False)
    params["bin_peaks"] = st.selectbox("bin_peaks", ["auto", "True", "False"])
    params["num_RT_bins"] = st.number_input("num_RT_bins", 10, 100, 50, 10)
    params["num_mz_bins"] = st.number_input("num_mz_bins", 10, 100, 50, 10)
    if params["bin_peaks"] == "True":
        params["bin_peaks"] = True
    elif params["bin_peaks"] == "False":
        params["bin_peaks"] = False
    return params

def get_Spectrum_params():
    params = {}
    params["ion_mobility"] = st.checkbox("ion_mobility")
    params["annotate_mz"] = st.checkbox("annotate_mz")
    params["annotate_ions"] = st.checkbox("annotate_ions")
    params["annotate_sequence"] = st.checkbox("annotate_sequence")
    params["mirror_spectrum"] = st.checkbox("mirror_spectrum")
    params["relative_intensity"] = st.checkbox("relative_intensity", False)
    params["custom_peak_color"] = st.checkbox("custom_peak_color")
    params["custom_annotation_text"] = st.checkbox("custom_annotation_text")
    params["custom_annotation_color"] = st.checkbox("custom_annotation_color")
    return params

if "exp_df" not in st.session_state:
    load_demo_mzML()

with st.sidebar:
    demo = st.selectbox("select demo", ["MSExperiment", "MSSpectrum"])

tabs = st.tabs(["📊 **Figure**", "📂 **Data**", "📑 **API docs**"])
if demo == "MSExperiment":
    with st.sidebar:
        st.markdown("**MSExperiment Parameters**")
        params = get_MSExperiment_params()
        st.markdown("**Common Parameters**")
        common_params = get_common_parameters()
    if common_params["engine"] != "PLOTLY":
        fig = plotMSExperiment(st.session_state.exp_df, **common_params, **params)
        with tabs[0]:
            display_fig(fig, common_params["engine"])
    else:
        with tabs[0]:
            st.info(
                "💡 Zoom in to reveals more details, the plot will update automatically."
            )
        df = st.session_state.exp_df
        if "msexp_selection" in st.session_state:
            points = st.session_state.msexp_selection.selection.points
            box = st.session_state.msexp_selection.selection.box
            if box:
                df = st.session_state.exp_df.copy()
                df = df[df["RT"] > box[0]["x"][0]]
                df = df[df["mz"] > box[0]["y"][1]]
                df = df[df["mz"] < box[0]["y"][0]]
                df = df[df["RT"] < box[0]["x"][1]]
        fig = plotMSExperiment(df, **common_params, **params)
        with tabs[0]:
            st.plotly_chart(
                fig,
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

    with tabs[1]:
        st.dataframe(st.session_state.exp_df)
    with tabs[2]:
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

    with st.sidebar:
        st.markdown("**Spectrum Parameters**")
        params = get_Spectrum_params()
        st.markdown("**Common Parameters**")
        common_params = get_common_parameters()
    fig = plotSpectrum(
        spec,
        spec,
        **params,
        **common_params
    )
    with tabs[0]:
        display_fig(fig, common_params["engine"])
    with tabs[1]:
        st.dataframe(spec)
    with tabs[2]:
        st.write(plotSpectrum)
