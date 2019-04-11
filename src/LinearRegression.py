import numpy as np

from src.Model import Model
from src.scoring import R2


class LinearRegression(Model):

    def __init__(self):
        """
        Multiple linear regression model that extends the abstract Model class
        """
        super().__init__()
        self._betas = None

    def fit(self, x, y):
        """ Fits the model to the training data by calculating the coefficients
        Overrides abstract fit method in abstract Model class

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
        """ Once the model has been fit, this method returns model estimates
        Overrides abstract predict method in abstract Model class

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
        """ Calculates R squared score for trained model
        Overrides abstract score method in abstract Model class

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
        return R2(y, self.predict(x))

    def get_betas(self):
        """ Gets model coefficients

        Returns
        -------
        numpy array or None
            if model fit - numpy array of coefficients
            if model not fit - None
        """
        return self._betas

    @staticmethod
    def _add_intercept_col(x):
        """ Adds an intercept column to the data ingested by the model (independent variable(s))

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
        """ Called by fit() to calculate the coefficients for the linear regression

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
