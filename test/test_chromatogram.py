"""
test/plotting/test_matplotlib
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import matplotlib
matplotlib.use("Agg")  # Use headless backend to avoid tk errors

import pytest
import pandas as pd
from pyopenms_viz._core import SpectrumPlot

@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(by="Annotation"),
        dict(by="Annotation", relative_intensity=True),
        dict(xlabel="RETENTION", ylabel="INTENSITY"),
        dict(grid=False),
    ],
)
def test_chromatogram_plot(chromatogram_data, snapshot, kwargs):
    out = chromatogram_data.plot(
        x="rt", y="int", kind="chromatogram", show_plot=False, **kwargs
    )

    # apply tight layout to matplotlib to ensure not cut off
    if pd.options.plotting.backend == "ms_matplotlib":
        fig = out.get_figure()
        fig.tight_layout()

    assert snapshot == out


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(by="Annotation"),
    ],
)
def test_chromatogram_with_annotation(
    chromatogram_data, chromatogram_features, snapshot, kwargs
):
    out = chromatogram_data.plot(
        x="rt",
        y="int",
        kind="chromatogram",
        annotation_data=chromatogram_features,
        show_plot=False,
        **kwargs,
    )
    # apply tight layout to maplotlib to ensure not cut off
    if pd.options.plotting.backend == "ms_matplotlib":
        fig = out.get_figure()
        fig.tight_layout()

    assert snapshot == out

# Dummy backend to test missing peptide sequence implementation
class DummySpectrumPlot(SpectrumPlot):
    def __init__(self, data):
        # Instead of calling super().__init__(data) (which verifies required columns),
        # we manually set minimal attributes needed to pass initialization.
        self.data = data.copy()
        # Set dummy column values so that _verify_column doesn't raise errors.
        self.x = "dummy"
        self.y = "dummy"
        self.by = None
        # Provide a dummy config with display_peptide_sequence enabled.
        self._config = type("dummy_config", (), {"display_peptide_sequence": True})()
        
    def plot(self):
        pass

    def load_config(self, **kwargs):
        pass

    def _check_and_aggregate_duplicates(self):
        pass

    def _prepare_data(self, data):
        return data

    def _create_tooltips(self, entries, index):
        return {}, {}

    def _get_annotations(self, data, x, y):
        return [], [], [], []

    def convert_for_line_plots(self, data, x, y):
        return data

    def _get_colors(self, data, kind):
        return "blue"

    def get_line_renderer(self, data, by=None, color=None, config=None):
        # Return dummy object with required methods.
        class DummyRenderer:
            def generate(self, tooltips, custom_hover_data):
                # Return a dummy canvas with a gca() method.
                class DummyAxes:
                    def gca(self):
                        return self
                return DummyAxes()
            def _add_annotations(self, canvas, texts, xs, ys, colors):
                pass
        return DummyRenderer()

    # Provide stubs for all abstract methods from SpectrumPlot:
    def _add_bounding_box_drawer(self):
        pass

    def _add_bounding_vertical_drawer(self):
        pass

    def _add_legend(self, legend):
        pass

    def _add_tooltips(self, tooltips, custom_hover_data):
        pass

    def _create_figure(self):
        return None

    def _load_extension(self):
        pass

    def _modify_x_range(self, x_range, padding=None):
        pass

    def _modify_y_range(self, y_range, padding=None):
        pass

    def _update_plot_aes(self):
        pass

    def generate(self, tooltips, custom_hover_data):
        return None

    def get_scatter_renderer(self, *args, **kwargs):
        return None

    def get_vline_renderer(self, *args, **kwargs):
        return None

    def plot_x_axis_line(self, canvas, line_width=2):
        pass

    def show_default(self):
        pass

    # Abstract method: raise error since it's not supported in this dummy backend.
    def plot_peptide_sequence(self, ax, data):
        raise NotImplementedError("Dummy backend does not support peptide sequence plotting")