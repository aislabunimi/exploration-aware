import matplotlib.pyplot as plt


def plot_history(hist):
    """
    Plot the training and validation loss and accuracy over epochs.
    """
    plt.figure(figsize=(12, 4))

    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(hist["train_loss"], label="Train Loss")
    plt.plot(hist["validation_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    # Plot each metric in the metrics dictionary
    plt.subplot(1, 2, 2)
    for metric_name, metric_values in hist["metrics"].items():
        plt.plot(metric_values, label=metric_name)
    plt.title("Model Metrics")
    plt.ylabel("Value")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.show()
