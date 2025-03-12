import matplotlib.pyplot as plt
import pandas as pd
from pyopenms_viz._matplotlib.core import MATPLOTLIBSpectrumPlot
from pyopenms_viz._config import SpectrumConfig

# Create a sample spectrum data
data = pd.DataFrame({
    "m/z": [100, 200, 300, 400, 500],
    "intensity": [10, 20, 30, 40, 50],
})

# Create a SpectrumConfig object with peptide sequence display enabled
config = SpectrumConfig(
    display_peptide_sequence=True,  # Enable peptide sequence display
    peptide_sequence_fontsize=14,  # Custom font size
    peptide_sequence_color="blue",  # Custom text color
    highlight_color="orange",  # Custom highlight color
    highlight_alpha=0.5,  # Custom transparency
    x="m/z",  # Set the x attribute
    y="intensity"  # Set the y attribute
)

# Create a MATPLOTLIBSpectrumPlot object
plot = MATPLOTLIBSpectrumPlot(data, config=config)

# Generate the plot
fig, ax = plt.subplots()
plot.canvas = ax  # Set the canvas for the plot

# Add the peptide sequence and matched fragments
peptide_sequence = "PEPTIDE"
matched_fragments = [(0, 2), (4, 6)]  # Highlight "PEP" and "TID"
plot.add_peptide_sequence(peptide_sequence, matched_fragments)

# Plot the spectrum (dummy data for demonstration)
ax.plot(data["m/z"], data["intensity"], color="black", label="Spectrum")

# Customize the plot
ax.set_xlabel("m/z")
ax.set_ylabel("Intensity")
ax.set_title("Mass Spectrum with Peptide Sequence")
ax.legend()

# Save the plot as an image
output_path = "spectrum_with_peptide_sequence.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Plot saved successfully at: {output_path}")