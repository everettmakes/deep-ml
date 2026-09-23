import numpy as np
import scipy.stats as stat

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    mean = float(np.mean(data))
    median = float(np.median(data))
    mode = float(stat.mode(data).mode)
    variance = float(np.var(data))
    standard_deviation = float(np.std(data))
    percentile_25th = float(np.percentile(data, 25))
    percentile_50th = float(np.percentile(data, 50))
    percentile_75th = float(np.percentile(data, 75))



    
    return {'mean': mean, 'median': median, 'mode': mode, 'variance': variance, 'standard_deviation': standard_deviation, '25th_percentile': percentile_25th,'50th_percentile': percentile_50th , '75th_percentile': percentile_75th, 'interquartile_range': percentile_75th - percentile_25th}