def plot(ax, data, x=0.5, y=0.5, spacing=0.06, fontsize="xx-large", fontsize_frag="medium", frag_len=0.06):
    """
    Plot peptide sequence with matched fragments indicated.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    data : pandas.DataFrame
        The spectrum dataframe.
    x : float, optional
        The center horizontal position of the peptide sequence.
    y : float, optional
        The center vertical position of the peptide sequence.
    spacing : float, optional
        The horizontal spacing between amino acids.
    fontsize : str, optional
        The font size of the amino acids.
    fontsize_frag : str, optional
        The font size of the fragment annotations.
    frag_len : float, optional
        The length of the fragment lines.
    """
    sequence = data["sequence"].iloc[0]
    n_residues = len(sequence)

    # Remap `x` position to be the left edge of the peptide.
    x = x - n_residues * spacing / 2 + spacing / 2

    # Plot the amino acids in the peptide.
    for i, aa in enumerate(sequence):
        ax.text(
            *(x + i * spacing, y, aa),
            fontsize=fontsize,
            ha="center",
            transform=ax.transAxes,
            va="center",
        )
    # Indicate matched fragments.
    for annot, color in zip(data["ion_annotation"], data["color_annotation"]):
        ion_type = annot[0]
        ion_i = int(i) if (i := annot[1:].rstrip("+")) else 1
        x_i = x + spacing / 2 + (ion_i - 1) * spacing

        # Length of the fragment line.
        if ion_type in "ax":
            y_i = 2 * frag_len
        elif ion_type in "by":
            y_i = frag_len
        elif ion_type in "cz":
            y_i = 3 * frag_len
        else:
            # Ignore unknown ion types.
            continue

        # N-terminal fragmentation.
        if ion_type in "abc":
            xs = [x_i, x_i, x_i - spacing / 2]
            ys = [y, y + y_i, y + y_i]
            nterm = True
        # C-terminal fragmentation.
        elif ion_type in "xyz":
            xs = [x_i + spacing / 2, x_i, x_i]
            ys = [y - y_i, y - y_i, y]
            nterm = False
        else:
            # Ignore unknown ion types.
            continue

        ax.plot(
            xs, ys, clip_on=False, color=color, transform=ax.transAxes
        )

        ax.text(
            x_i,
            y + (1.05 if nterm else -1.05) * y_i,
            annot,
            color=color,
            fontsize=fontsize_frag,
            ha="right" if nterm else "left",
            transform=ax.transAxes,
            va="top" if not nterm else "bottom",
        )
