# //// filepath: c:\Users\sayus\OneDrive\Desktop\OpenMS\pyopenms_viz2\EX.py
import matplotlib.pyplot as plt
import pandas as pd

# Adjust the import if your plotting class is located elsewhere.
from pyopenms_viz._matplotlib.core import MATPLOTLIBSpectrumPlot

# Read the TSV file containing the spectrum data.
df = pd.read_csv("test/test_data/TestSpectrumDf.tsv", sep="\t")

# Modify some fragment annotations for testing.
df.loc[0, "ion_annotation"] = "a3+"
df.loc[2, "ion_annotation"] = "c2+"

print("Ion Annotations:")
print(df["ion_annotation"])

# Create a matplotlib figure and axes.
fig, ax = plt.subplots()

# Create an instance of the matplotlib spectrum plot.
# IMPORTANT: Pass the required 'x' and 'y' column names from your dataframe.
plotter = MATPLOTLIBSpectrumPlot(data=df, x="mz", y="intensity")
# Then assign the axes.
plotter.ax = ax

# Use the peptide sequence from the TSV; here we take the first row.
peptide_seq = df["sequence"].iloc[0]

# Prepare matched fragments using the 'ion_annotation' and 'color_annotation' columns.
matched_fragments = list(zip(df["ion_annotation"], df["color_annotation"]))

# Call the plotting method to render the peptide sequence with fragments.
plotter.plot_peptide_sequence(
    peptide_sequence=peptide_seq,
    matched_fragments=matched_fragments,
    x=0.5,
    y=0.95,
    spacing=0.05,
    fontsize=12,
    fontsize_frag=10,
    frag_len=0.05
)

# Optionally, set axis limits and ticks if needed.
ax.set_ylim(-0.04, 0.04)
ax.set_yticks([-0.04, -0.02, 0, 0.02, 0.04])

# Save the generated plot to a PNG file.
plt.savefig("sequenceplot.png", dpi=300)
plt.close()