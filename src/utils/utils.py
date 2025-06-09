from ..core.constants import EPOCHS, BATCH_SIZE, SAVE_FOLDER, OPTIM, CLASSES
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=float, default=BATCH_SIZE)
    parser.add_argument("--optim", type=str, choices=["sgd", "adam"], default=OPTIM["alg"])
    parser.add_argument("--checkpoint", type=str, default=False)
    parser.add_argument("--mt", type=float, default=OPTIM["momentum"])
    parser.add_argument("--lr", type=float, default=OPTIM["lr"])
    parser.add_argument("--es_min_delta", type=str, default=0,)
    parser.add_argument("--es_patience", type=int, default=4)
    parser.add_argument("--save_path", type=str, default=f"{SAVE_FOLDER}/hand_draw_model.pth")
    args = parser.parse_args()
    return args

def plot_metrisc_tensorboard(writer, f_mtrs):
    """
        Trực quan hóa accuracy và loss trong quá trình huấn luyện và validation.
        Args:
            f_mtrs (dict): danh sách các accuracy và loss qua các lần train.
            writer (SummaryWriter): đối tượng ghi log cho TensorBoard.
        Returns:
            None
    """
    fig, ax = plt.subplots(1, 2, figsize=(8, 6))
    # for loop
    for i in range(2):
        # template for loss and accuracy
        temp = "loss" if i == 0 else "accuracy"
        # plot loss
        ax[i].plot(f_mtrs["loss"] if i == 0 else f_mtrs["accuracy"], label=f'training {temp}', color='blue') # training phase
        ax[i].plot(f_mtrs["val_loss"] if i == 0 else f_mtrs["val_accuracy"], label=f'validation {temp}', color='orange') # validation phase
        ax[i].set_title(f'{temp}')
        ax[i].legend(loc='lower right')
    # # set title
    fig.suptitle('Visualization Metrics', fontsize=16)
    writer.add_figure("Visualization Metrics", fig, global_step=EPOCHS+1)
    fig.savefig(f"plots/metrics_plot.png")
    # show plot
    plt.close(fig)

def plot_confusion_matrix_tensorboard(writer, all_labels, all_preds, epoch):
    """
        Confusion matrix (ma trận nhẫm lẫn)
        Args:
            writer: Writer để ghi dữ liệu vào tensorboard
            all_labels: các labels
            all_preds: các dự đoán
            epoch: epoch hiện tại
        Returns:
            None
        
    """
    
    cm = confusion_matrix(all_labels, all_preds)
    figure = plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(list(CLASSES.keys())))
    plt.xticks(tick_marks, list(CLASSES.keys()), rotation=45)
    plt.yticks(tick_marks, list(CLASSES.keys()))

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    figure.savefig(f"plots/confusion_matrix_plot.png")
    writer.add_figure('confusion_matrix', figure, epoch)