from typing import Dict

from torch import nn

from pkg.interface.config import IConfig
from pkg.utils.other_utils.logs import init_logger

logger = init_logger("BASE_MODEL")


class BaseModule(nn.Module, IConfig):
    """Base module class for neural network models.

    Provides core functionality for model configuration and initialization.
    Subclasses implement specific model architectures.

    Args:
        config (Dict): Configuration dictionary containing model parameters
        *args: Variable length argument list
        **kwargs: Arbitrary keyword arguments

    """

    def __init__(self, config: Dict, *args, **kwargs) -> None:
        self._prefix_name = "base_module"
        if "prefix_name" in config:
            self._prefix_name = config["prefix_name"]
        elif "prefix_name" in kwargs:
            self._prefix_name = kwargs.pop("prefix_name")

        super(BaseModule, self).__init__(*args, **kwargs)

    @property
    def prefix_name(self) -> str:
        return self._prefix_name

    def _init_graph(self) -> None:
        """Initialize the model computation graph.

        Must be implemented by subclasses to define model architecture.

        Raises:
            NotImplementedError: If subclass does not implement this method
        """
        raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required 'init_graph' function")

    def get_config(self) -> Dict:
        """Get model configuration.

        Returns:
            Dict: Configuration dictionary containing model parameters
        """
        return {
            "prefix_name": self._prefix_name,
        }
