from matplotlib import pyplot as plt

def loss_graphic_creation(epochs, loss_value, val_loss_value):
    epochs = range(1, epochs + 1)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_value, "bo", label="Train loss value")
    plt.plot(epochs, val_loss_value, "b", label="Val loss value")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()


def accuracy_graphic_creation(epochs, accuracy, val_accuracy):
    epochs = range(1, epochs + 1)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, "bo", label="Train accuracy value")
    plt.plot(epochs, val_accuracy, "b", label="Val accuracy value")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def draw_graphics(epochs, loss_value, val_loss_value, accuracy, val_accuracy):
    plt.figure(figsize=(50, 50))
    loss_graphic_creation(epochs, loss_value, val_loss_value)
    accuracy_graphic_creation(epochs, accuracy, val_accuracy)