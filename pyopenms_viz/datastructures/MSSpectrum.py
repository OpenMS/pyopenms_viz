"""
Schema definition for mass spectrometry spectrum.
"""

REQUIRED_SPECTRUM_DATAFRAME_COLUMNS = {
    "native_id": "String column representing the native identifier (i.e. scan number) of the spectrum.",
    "mz": "Numeric column representing the mass-to-charge ratio (m/z) values of the peaks in the mass spectrum.",
    "intensity": "Numeric column representing the intensity (abundance) of the peaks in the mass spectrum.",
    "ms_level": "Integer column indicating the MS level (1 for MS1, 2 for MS2, etc.)."
}

OPTIONAL_METADATA_SPECTRUM_DATAFRAME_COLUMNS = {
    "sequence": "String column representing the peptide sequence.",
    "modified_sequence": "String column representing the modified peptide sequence.",
    "precursor_mz": "Numeric column representing the mass-to-charge ratio (m/z) of the precursor ion.",
    "precursor_charge": "Integer column representing the charge state of the precursor ion.",
    "product_mz": "Numeric column representing the mass-to-charge ratio (m/z) of the product ion.",
    "product_charge": "Integer column representing the charge state of the product ion.",
    "annotation": "String column representing the annotation of the spectrum, such as the fragment ion series."
}