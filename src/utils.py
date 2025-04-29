import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
import tensorflow as tf
from yt_dlp import YoutubeDL

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


def create_confusion_matrix(dataset, model, class_names):
    """
    Creates and displays a confusion matrix for the given dataset and model.
    
    Args:
        dataset: TensorFlow dataset containing images and labels
        model: Trained Keras model
        class_names: List of class names
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns  # You may need to install this: pip install seaborn
    
    # Initialize empty lists to store true labels and predictions
    y_true = []
    y_pred = []
    
    # Iterate through the dataset
    print("Collecting predictions...")
    for images, labels in dataset:
        # Get predictions
        predictions = model.predict(images, verbose=0)
        
        # Convert one-hot encoded labels to class indices
        true_classes = tf.argmax(labels, axis=1).numpy()
        pred_classes = tf.argmax(predictions, axis=1).numpy()
        
        # Append to lists
        y_true.extend(true_classes)
        y_pred.extend(pred_classes)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    return y_true, y_pred


def download_mp4(url, output_path='.'):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        # no need for merge_output_format since /mp4 forces ffmpeg merge
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])