import abc
from typing import Dict


class IDataPreparationDataset(abc.ABC):
    """Base abstract class for dataset preparation.

    Defines interface for data generation and statistics computation.
    Subclasses implement specific preparation logic.

    Methods:
        process(): Main dataset preparation workflow
        _data_generation(): Generate dataset
        _data_stats(): Compute dataset statistics
        get_config(): get dataset config
    """

    @abc.abstractmethod
    def process(self) -> None:
        """Execute dataset preparation workflow.

        Subclasses implement specific preparation steps.
        """
        raise NotImplementedError("Subclasses must implement the process method.")

    @abc.abstractmethod
    def _data_generation(self) -> None:
        """Generate dataset.

        Subclasses implement data generation logic.
        """
        raise NotImplementedError("Subclasses must implement the data_generation method.")

    @abc.abstractmethod
    def _data_stats_generation(self) -> None:
        """Compute dataset statistics.

        Subclasses implement statistics computation.
        """
        raise NotImplementedError("Subclasses must implement the data_stats_generation method.")

    @property
    @abc.abstractmethod
    def get_config(self) -> Dict:
        """Get class config.

        Subclasses implement config fetch.
        """
        raise NotImplementedError("Subclasses must implement the get_config method.")


class ITrainDataset(abc.ABC):
    @abc.abstractmethod
    def __len__(self) -> int:
        """Get dataset size.

        Returns:
            int: Number of samples in dataset
        """
        raise NotImplementedError("Subclasses must implement __len__ method")

    @abc.abstractmethod
    def _validation(self) -> None:
        raise NotImplementedError("Subclasses must implement _validation method")

    @abc.abstractmethod
    def get_head_inputs(self, batch_size: int) -> Dict:
        """Get model head inputs for architecture visualization.

        Args:
            batch_size (int): Number of samples to generate

        Returns:
            Dict: Model head inputs
        """
        raise NotImplementedError("Subclasses must implement get_head_inputs method")
