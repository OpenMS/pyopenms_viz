from enum import Enum, auto
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle


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
            self.color_cycle = cycle(self.color_blind_friendly_map.values())
        else:
            if isinstance(colormap, str):
                if colormap.lower() == "grayscale":
                    hex_colors = self._get_n_grayscale_colors(n)
                    self.color_cycle = cycle(hex_colors)
                else:
                    cmap = plt.get_cmap(colormap, n)
                    colors = cmap(np.linspace(0, 1, n))
                    hex_colors = [
                        "#{:02X}{:02X}{:02X}".format(
                            int(r * 255), int(g * 255), int(b * 255)
                        )
                        for r, g, b, _ in colors
                    ]
                    self.color_cycle = cycle(hex_colors)
            else:
                self.color_cycle = cycle(colormap)

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
