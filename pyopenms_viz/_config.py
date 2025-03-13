from abc import ABC, ABCMeta
from dataclasses import dataclass, field, asdict, fields
from typing import Tuple, Literal, Dict, Any, Union, Iterator
from copy import deepcopy
from ._misc import ColorGenerator, MarkerShapeGenerator
import pandas as pd


@dataclass(kw_only=True)
class BaseConfig(ABC):

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """
        Convert a dictionary to a LegendConfig instance.

        Args:
        legend_dict (Dict[str, Any]): Dictionary containing legend configuration.

        Returns:
        BaseConfig: An child class from BaseConfig
        """

        config = asdict(cls())

        # Update with provided values
        config.update(config_dict)

        # Create and return the LegendConfig instance
        return cls(**config)

    def update_none_fields(self, other: "BaseConfig") -> None:
        """
        Update only the fields that are None with values from another BaseConfig object.

        Args:
        other (BaseConfig): Another BaseConfig object containing values to update.
        """
        for field_obj in fields(other):
            field_name = field_obj.name
            if getattr(self, field_name) is None:
                setattr(self, field_name, getattr(other, field_name))

    def update(self, **kwargs):
        """Update the dataclass fields with kwargs."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(
                    f"{key} is not a valid attribute for {self.__class__.__name__}"
                )


@dataclass(kw_only=True)
class LegendConfig(BaseConfig):
    """
    Configuration for the legend in a plot.

    Args:
        loc (str): Location of the legend. Default is "right".
        orientation (str): Orientation of the legend. Default is "vertical".
        title (str): Title of the legend. Default is "Legend".
        fontsize (int): Font size of the legend text. Default is 10.
        show (bool): Whether to show the legend. Default is True.
        onClick (Literal["hide", "mute"]): Action on legend click. Only valid for Bokeh. Default is "mute".
        bbox_to_anchor (Tuple[float, float]): Fine control for legend positioning in Matplotlib. Default is (1.2, 0.5).

    Returns:
        LegendConfig: An instance of LegendConfig.
    """

    loc: str = "right"
    orientation: str = "vertical"
    title: str = "Legend"
    fontsize: int = 10
    show: bool = True
    onClick: Literal["hide", "mute"] = (
        "mute"  # legend click policy, only valid for bokeh
    )
    bbox_to_anchor: Tuple[float, float] = (
        1.2,
        0.5,
    )  # for fine control legend positioning in matplotlib

    @staticmethod
    def _matplotlibLegendLocationMapper(loc):
        """
        Maps the legend location to the matplotlib equivalent
        """
        loc_mapper = {
            "right": "center right",
            "left": "center left",
            "above": "upper center",
            "below": "lower center",
        }
        return loc_mapper[loc]

    @classmethod
    def from_dict(cls, legend_dict: Dict[str, Any]) -> "LegendConfig":
        """
        Convert a dictionary to a LegendConfig instance.

        Args:
        legend_dict (Dict[str, Any]): Dictionary containing legend configuration.

        Returns:
        LegendConfig: An instance of LegendConfig with the specified settings.
        """
        # Create a dictionary with default values
        config = super().from_dict(legend_dict)

        # Ensure onClick is a valid value
        if config.onClick not in ["hide", "mute"]:
            config.onClick = "mute"

        return config
