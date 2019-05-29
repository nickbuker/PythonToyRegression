import numpy as np


def rss(y: np.ndarray, y_hat: np.ndarray) -> float:
    """ Residual Sum of Squares

    Parameters
    ----------
    y
        y values
    y_hat
        y values estimated by model

    Returns
    -------
    float
        residual sum of squares
    """
    return np.sum((y - y_hat) ** 2)


def tss(y: np.ndarray) -> float:
    """ Total Sum of Squares

    Parameters
    ----------
    y
        y values

    Returns
    -------
    float
        total sum of squares
    """
    return np.sum((y - np.mean(y)) ** 2)


def ess(y: np.ndarray, y_hat: np.ndarray) -> float:
    """ Explained Sum of Squares

    Parameters
    ----------
    y
        y values
    y_hat
        y values estimated by model

    Returns
    -------
    float
        explained sum of squares
    """
    return np.sum((y_hat - np.mean(y)) ** 2)


def r2(y: np.ndarray, y_hat: np.ndarray) -> float:
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
    return 1 - (rss(y, y_hat) / tss(y))