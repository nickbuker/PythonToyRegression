import numpy as np
from abc import ABC, abstractmethod


class Model(ABC):

    def __init__(self):
        """
        Model abstract base class
        """
        super().__init__()

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """ Abstract method for training model

        Parameters
        ----------
        x
            data on which to train the model (independent variable(s))
        y
            data on which to train the model (dependent variable)

        Returns
        -------
        None
        """
        return

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """ Abstract method for making predictions with a trained model

        Parameters
        ----------
        x
            data on which to make estimates (independent variable(s))

        Returns
        -------
        numpy array
            y-hat (dependent variable estimates)
        """
        return

    @abstractmethod
    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """ Abstract method for scoring a trained model

        x
            data on which to score the model (independent variable(s))
        y
            data on which to score the model (dependent variable)

        Returns
        -------
        float
            score for the model
        """
        return
