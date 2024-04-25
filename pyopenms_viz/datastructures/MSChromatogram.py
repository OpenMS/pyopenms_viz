"""
Schema definition for mass spectrometry chromatogram.
"""

REQUIRED_CHROMATOGRAM_DATAFRAME_COLUMNS = {
    "mz": "Numeric column representing the mass-to-charge ratio (m/z) of the extracted retention time point.",
    "time": "Numeric column representing the retention time (in minutes) of the chromatographic peaks.",
    "intensity": "Numeric column representing the intensity (abundance) of the signal at each time point.",
    "ms_level": "Integer column indicating the MS level (1 for MS1, 2 for MS2, etc.)."
}

OPTIONAL_METADATA_CHROMATOGRAM_DATAFRAME_COLUMNS = {
    "sequence": "String column representing the peptide sequence.",
    "modified_sequence": "String column representing the modified peptide sequence.",
    "precursor_mz": "Numeric column representing the mass-to-charge ratio (m/z) of the precursor ion.",
    "precursor_charge": "Integer column representing the charge state of the precursor ion.",
    "product_mz": "Numeric column representing the mass-to-charge ratio (m/z) of the product ion.",
    "product_charge": "Integer column representing the charge state of the product ion.",
    "annotation": "String column representing the annotation of the spectrum, such as the fragment ion series."
}

OPTIONAL_FEATURE_CHROMATOGRAM_DATAFRAME_COLUMNS = {
    "rt_apex": "Numeric column representing the retention time (in minutes) of the peak apex.",
    "left_width": "Numeric column representing the width of the peak on the left side of the apex.",
    "right_width": "Numeric column representing the width of the peak on the right side of the apex.",
    "area": "Numeric column representing the area under the peak.",
    "q_value": "Numeric column representing the q-value of the peak."
}