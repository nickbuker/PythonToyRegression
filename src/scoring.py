import numpy as np


def RSS(y, y_hat):
    """ Residual Sum of Squares

    Parameters
    ----------
    y : numpy array
        y values
    y_hat : numpy array
        y values estimated by model

    Returns
    -------
    float
        residual sum of squares
    """
    return np.sum((y - y_hat) ** 2)


def TSS(y):
    """ Total Sum of Squares

    Parameters
    ----------
    y : numpy array
        y values

    Returns
    -------
    float
        total sum of squares
    """
    return np.sum((y - np.mean(y)) ** 2)


def ESS(y, y_hat):
    """ Explained Sum of Squares

    Parameters
    ----------
    y : numpy array
        y values
    y_hat : numpy array
        y values estimated by model

    Returns
    -------
    float
        explained sum of squares
    """
    return np.sum((y_hat - np.mean(y)) ** 2)


def R2(y, y_hat):
    """ R squared (coefficient of determination)

    Parameters
    ----------
    y : numpy array
        y values
    y_hat : numpy array
        y values estimated by model

    Returns
    -------
    float
        R squared
    """
    return 1 - (RSS(y, y_hat) / TSS(y))