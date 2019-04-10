import numpy as np

from src.Model import Model


class LinearRegression(Model):

    def __init__(self):
        """
        Multiple linear regression model that extends the abstract Model class
        """
        super().__init__()
        self._betas = None

    def fit(self, x, y):
        """
        TODO

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
        print('fit')
        x = self._add_intercept_col(x)
        self._find_betas(x, y)
        return

    def predict(self, x):
        """
        TODO

        Parameters
        ----------
        x : numpy array
            data on which to make estimates (independent variable(s))

        Returns
        -------
        numpy array
            y-hat (dependent variable estimates)
        """
        print('predict')
        x = self._add_intercept_col(x)
        return x.dot(self._betas)

    def score(self, x, y):
        """
        TODO

        Parameters
        ----------
        x : numpy array
            data on which to score the model (independent variable(s))
        y : numpy array
            data on which to score the model (dependent variable)

        Returns
        -------
        float
            R-squared score for the model
        """
        print('predict')
        return

    @staticmethod
    def _add_intercept_col(x):
        """
        TODO

        Parameters
        ----------
        x : numpy array
            data for the model (independent variable(s))

        Returns
        -------
        numpy array
            data for the model (independent variable(s)) with a column of 1's appended
        """
        intercept_col = np.ones(x.shape[0])
        return np.insert(x, 0, intercept_col, axis=1)

    def _find_betas(self, x, y):
        """
        TODO

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
        # TODO: implement QR and/or SVD solution
        # solve for betas using (X'X)^-1 X'Y
        self._betas = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
        return
