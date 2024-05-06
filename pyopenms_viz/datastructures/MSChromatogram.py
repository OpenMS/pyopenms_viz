"""
Schema definition for mass spectrometry chromatogram. This is used to store any chromatogram object
"""

REQUIRED_CHROMATOGRAM_DATAFRAME_COLUMNS = {
    "time": "Numeric column representing the retention time (in seconds) of the chromatographic peaks.",
    "intensity": "Numeric column representing the intensity (abundance) of the signal at each time point.",
}

OPTIONAL_METADATA_CHROMATOGRAM_DATAFRAME_COLUMNS = {
    "native_id" : "Chromatogram id, necessary if multiple chromatograms are in the same dataframe."
    "chromatogram_type": "Type of chromatogram must be one of: MASS_CHROMATOGRAM, TOTAL_ION_CURRENT_CHROMATOGRAM,  SELECTED_ION_CURRENT_CHROMATOGRAM, BASEPEAK_CHROMATOGRAM, SELECTED_ION_MONITORING_CHROMATOGRAM, SELECTED_REACTION_MONITORING_CHROMATOGRAM, ELECTROMAGNETIC_RADIATION_CHROMATOGRAM, ABSORPTION_CHROMATOGRAM, EMISSION_CHROMATOGRAM"
    "ms_level": "Integer column indicating the MS level (1 for MS1, 2 for MS2, etc.)."
    "sequence": "String column representing the peptide sequence.",
    "modified_sequence": "String column representing the modified peptide sequence. Modification can be represented using the UniMod ontology, either with the UniMod Accession (UniMod: 21) or with the UniMod Codename (Phospho).",
    "precursor_mz": "Numeric column representing the mass-to-charge ratio (m/z) of the precursor ion.",
    "precursor_charge": "Integer column representing the charge state of the precursor ion.",
    "product_mz": "Numeric column representing the mass-to-charge ratio (m/z) of the product ion.",
    "product_charge": "Integer column representing the charge state of the product ion.",
    "annotation": "String column representing the annotation of the spectrum, such as the fragment ion series."
}

