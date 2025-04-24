import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)

def plot_results(metrics, title=None, ylabel=None, metric_name=None, color=None):
    """
    Plot the results of the training process.

    Parameters:
    ----------
    metrics: list of lists or tuples containing the metrics to plot
    title: title of the plot
    ylabel: y-axis label
    metric_name: names of the metrics to plot
    color: list of colors for the metrics
    """

    fig, ax = plt.subplots(figsize=(10, 4))

    if not (isinstance(metric_name, list) or isinstance(metric_name, tuple)):
        metrics = [metrics,]
        metric_name = [metric_name,]
        
    for idx, metric in enumerate(metrics):    
        ax.plot(metric, color=color[idx])
    
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.legend(metric_name)   
    plt.show(block=True)
    plt.close()