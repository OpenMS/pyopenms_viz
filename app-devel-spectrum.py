import streamlit as st


import pandas as pd
from pyopenms_viz._matplotlib.core import MATPLOTLIBSpectrumPlot, MATPLOTLIBPeakMapPlot
from pyopenms_viz._plotly.core import PLOTLYSpectrumPlot, PLOTLYPeakMapPlot
from pyopenms_viz._bokeh.core import BOKEHSpectrumPlot, BOKEHPeakMapPlot

import pandas as pd
from bokeh.plotting import figure, save
from bokeh.io import output_file
import streamlit.components.v1 as components


# Current streamlit version only supports bokeh 2.4.3
# See work around: https://github.com/streamlit/streamlit/issues/5858#issuecomment-1482042533
def use_file_for_bokeh(chart: figure, chart_height=500):
    output_file("bokeh_graph.html")
    save(chart)
    with open("bokeh_graph.html", "r", encoding="utf-8") as f:
        html = f.read()
    components.html(html, height=chart_height)


# Update the bokeh_chart method to use the file workaround
st.bokeh_chart = use_file_for_bokeh

df = pd.read_csv("test/test_data/TestSpectrumDf.tsv", sep="\t")

st.write(df)

tabs = st.tabs(["spectrum", "ion mobility spectrum"])
with tabs[0]:
    c1, c2 = st.columns(2)
    peak_color = c1.selectbox("peak_color", ["None"] + df.columns.tolist())
    annotation_color = c1.selectbox("annotation_color", ["None"] + df.columns.tolist())
    ion_annotation = c1.selectbox("ion_annotation", ["None"] + df.columns.tolist())
    sequence_annotation = c1.selectbox(
        "sequence_annotation", ["None"] + df.columns.tolist()
    )
    custom_annotation = c1.selectbox("custom_annotation", ["None"] + df.columns.tolist())
    by = c1.selectbox("by", ["None"] + df.columns.tolist())
    mirror_spectrum = c2.checkbox("mirror_spectrum", False)
    relative_intensity = c2.checkbox("relative_intensity", False)
    show_legend = c2.checkbox("show_legend", False)
    annotate_mz = c2.checkbox("annotate_mz", True)
    annotate_top_n_peaks = c2.number_input("annotate_top_n_peaks", 0, 100, 3, 1)
    kwargs = {
        "data": df,
        "x": "mz",
        "y": "intensity",
        "reference_spectrum": df,
        "show_plot": False,
        "relative_intensity": relative_intensity,
        "peak_color": peak_color,
        "annotation_color": annotation_color,
        "ion_annotation": ion_annotation,
        "sequence_annotation": sequence_annotation,
        "custom_annotation": custom_annotation,
        "annotate_mz": annotate_mz,
        "mirror_spectrum": mirror_spectrum,
        "by": None if by == "None" else by,
        "legend": {"show": show_legend},
        "annotate_top_n_peaks": annotate_top_n_peaks,
    }

    fig = PLOTLYSpectrumPlot(**kwargs).fig

    st.plotly_chart(fig)

    fig = MATPLOTLIBSpectrumPlot(**kwargs).superFig

    st.pyplot(fig)

    fig = BOKEHSpectrumPlot(**kwargs).fig

    st.bokeh_chart(fig)


with tabs[1]:
    kwargs = {
        "data": df,
        "x": "mz",
        "y": "ion_mobility",
        "z": "intensity",
        "show_plot": False,
        "relative_intensity": relative_intensity,
        "by": None if by == "None" else by,
        "legend": {"show": show_legend},
    }

    fig = PLOTLYPeakMapPlot(**kwargs).fig

    st.plotly_chart(fig)

    fig = MATPLOTLIBPeakMapPlot(**kwargs).superFig

    st.pyplot(fig)

    fig = BOKEHPeakMapPlot(**kwargs).fig

    st.bokeh_chart(fig)
