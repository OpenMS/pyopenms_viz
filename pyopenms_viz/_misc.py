from enum import Enum, auto
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from typing import Literal


class ColorGenerator:
    """
    A class that generates colors for plotting.

    Attributes:
        color_blind_friendly_map (dict): A dictionary that maps the enum values to their corresponding colors.

    Methods:
        __init__(self, colormap=None, n=None): Initializes a ColorGenerator object.
        __iter__(self): Returns the ColorGenerator object itself.
        __next__(self): Returns the next color in the color cycle.
        generate_colors(self, n=None): Generates a list of colors.
        _get_n_grayscale_colors(self, n: int) -> list: Returns n evenly spaced grayscale colors in hex format.
    """

    class Colors(Enum):
        """
        Enum class that defines color options.
        """

        BLUE = auto()
        RED = auto()
        LIGHTBLUE = auto()
        ORANGE = auto()
        PURPLE = auto()
        YELLOW = auto()
        DARKGRAY = auto()
        LIGHTGRAY = auto()

    color_blind_friendly_map = {
        Colors.BLUE: "#4575B4",
        Colors.RED: "#D73027",
        Colors.LIGHTBLUE: "#91BFDB",
        Colors.ORANGE: "#FC8D59",
        Colors.PURPLE: "#7B2C65",
        Colors.YELLOW: "#FCCF53",
        Colors.DARKGRAY: "#555555",
        Colors.LIGHTGRAY: "#BBBBBB",
    }

    def __init__(self, colormap=None, n=None):
        """
        Initializes a ColorGenerator object.

        Args:
            colormap (str or list, optional): The colormap to use for generating colors. Defaults to None.
            n (int, optional): The number of colors to generate. Defaults to None.
        """
        if colormap is None:
            colors = list(self.color_blind_friendly_map.values())
        else:
            if isinstance(colormap, str):
                if colormap.lower() == "grayscale":
                    colors = self._get_n_grayscale_colors(n)
                else:
                    cmap = plt.get_cmap(colormap, n)
                    colors = cmap(np.linspace(0, 1, n))
                    colors = [
                        "#{:02X}{:02X}{:02X}".format(
                            int(r * 255), int(g * 255), int(b * 255)
                        )
                        for r, g, b, _ in colors
                    ]
            else:
                colors = colormap
        if n is not None:
            colors = colors[:n]
        self.color_cycle = cycle(colors)

    def __iter__(self):
        """
        Returns the ColorGenerator object itself.
        """
        return self

    def __next__(self):
        """
        Returns the next color in the color cycle.
        """
        return next(self.color_cycle)

    def generate_colors(self, n=None):
        """
        Generates a list of colors.

        Args:
            n (int, optional): The number of colors to generate. Defaults to None.

        Returns:
            list: A list of colors.
        """
        if n is None:
            return next(self)
        else:
            return [next(self) for _ in range(n)]

    def _get_n_grayscale_colors(self, n: int) -> list:
        """
        Returns n evenly spaced grayscale colors in hex format.

        Args:
            n (int): The number of colors to generate.

        Returns:
            list: A list of grayscale colors in hex format.
        """
        hex_list = []
        for v in np.linspace(50, 200, n):
            hex_color = "#"
            for _ in range(3):
                hex_color += f"{int(round(v)):02x}"
            hex_list.append(hex_color)
        return hex_list


class MarkerShapeGenerator:
    """
    A class that generates colors for plotting.

    Attributes:
        color_blind_friendly_map (dict): A dictionary that maps the enum values to their corresponding colors.

    Methods:
        __init__(self, colormap=None, n=None): Initializes a ColorGenerator object.
        __iter__(self): Returns the ColorGenerator object itself.
        __next__(self): Returns the next color in the color cycle.
        generate_colors(self, n=None): Generates a list of colors.
    """

    class Shapes(Enum):
        """
        Enum class that defines marker shape options.
        """

        SQUARE = auto()
        CIRCLE = auto()
        DIAMOND = auto()
        CROSS = auto()
        X = auto()
        TRIANGLE = auto()
        DOWN = auto()
        LEFT = auto()
        RIGHT = auto()
        PENTAGON = auto()
        HEXAGON = auto()

    matplotlib_map = {
        Shapes.SQUARE: "s",
        Shapes.CIRCLE: "o",
        Shapes.DIAMOND: "d",
        Shapes.CROSS: "+",
        Shapes.X: "x",
        Shapes.TRIANGLE: "^",
        Shapes.DOWN: "v",
        Shapes.HEXAGON: "h",
    }

    plotly_map = {
        Shapes.SQUARE: "square",
        Shapes.CIRCLE: "circle",
        Shapes.DIAMOND: "diamond",
        Shapes.CROSS: "cross",
        Shapes.X: "x",
        Shapes.TRIANGLE: "triangle-up",
        Shapes.DOWN: "triangle-down",
        Shapes.HEXAGON: "hexagon",
    }

    bokeh_map = {
        Shapes.SQUARE: "square",
        Shapes.CIRCLE: "circle",
        Shapes.DIAMOND: "diamond",
        Shapes.CROSS: "cross",
        Shapes.X: "x",
        Shapes.TRIANGLE: "triangle",
        Shapes.DOWN: "inverted_triangle",
        Shapes.HEXAGON: "hex",
    }

    def __init__(
        self,
        shapes: list | None = None,
        engine: Literal["MATPLOTLIB", "PLOTLY", "BOKEH"] | None = None,
        n: int | None = None,
    ):
        """
        Initializes a MarkerShapeGenerator object.

        Args:
            shapes (list or None, optional): A pre-defined sequence of shapes. Defaults to None.
            engine (Literal["MATPLOTLIB", "PLOTLY", "BOKEH"] or None, optional): Plotting engine. Defaults to None.
            n (int or None, optional): The number of shapes to generate. Defaults to None.
        """
        if engine is None and shapes is None:
            raise ValueError("Pass either a list of shapes or an engine to initialize the MarkerShapeGenerator.")
        if shapes is not None:
            self.shape_cycle = cycle(shapes)
            return
        if engine == "MATPLOTLIB":
            shapes = list(self.matplotlib_map.values())
        elif engine == "PLOTLY":
            shapes = list(self.plotly_map.values())
        else:
            shapes = list(self.bokeh_map.values())
        if n is not None:
            shapes = shapes[:n]
        self.shape_cycle = cycle(shapes)
        return

    def __iter__(self):
        """
        Returns the MarkerShapeGenerator object itself.
        """
        return self

    def __next__(self):
        """
        Returns the next shape in the shape cycle.
        """
        return next(self.shape_cycle)


def sturges_rule(df, value):
    """
    Calculate the number of bins using Sturges' rule.
    
    Args:
        df (pd.DataFrame): A pandas DataFrame containing the data.
        value (str): The column name of the data.
        
    Returns:
        int: The number of bins.
    """
    n = len(df[value])
    num_bins = int(np.ceil(1 + np.log2(n)))
    return num_bins

def freedman_diaconis_rule(df, value):
    """
    Calculate the number of bins using the Freedman-Diaconis rule.
    
    Args:
        df (pd.DataFrame): A pandas DataFrame containing the data.
        value (str): The column name of the data.
        
    Returns:
        int: The number of bins.
    """
    # Calculate IQR
    Q1 = df[value].quantile(0.25)
    Q3 = df[value].quantile(0.75)
    IQR = Q3 - Q1

    # Number of observations
    n = len(df)

    # Calculate bin width using the Freedman-Diaconis rule
    bin_width = 2 * IQR / (n ** (1/3))

    # Calculate the number of bins
    num_bins = int((df[value].max() - df[value].min()) / bin_width)
    return num_bins