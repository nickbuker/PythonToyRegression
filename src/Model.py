from abc import ABC, abstractmethod


class Model(ABC):

    def __init__(self):
        """
        Model abstract base class
        """
        super().__init__()

    @abstractmethod
    def fit(self, x, y):
        """

        Parameters
        ----------
        x : numpy array
            data on which to train the model (independent variable(s))
        y : numpy array
            data on which to train the model (dependent variable)

        Returns
        -------
        None
        """
        return

    @abstractmethod
    def predict(self, x):
        """

        Parameters
        ----------
        x : numpy array
            data on which to make estimates (independent variable(s))

        Returns
        -------
        numpy array
            y-hat (dependent variable estimates)
        """
        return

    @abstractmethod
    def score(self, x, y):
        """

        x : numpy array
            data on which to score the model (independent variable(s))
        y : numpy array
            data on which to score the model (dependent variable)

        Returns
        -------
        float
            score for the model
        """
        return
