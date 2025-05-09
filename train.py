from src.core.constants import DEVICE, CLASSES
from torch import save, no_grad, cat
from torch.utils.data import DataLoader, random_split
from src.utils.utils import get_arguments
from src.core.dataset import HandDrawDataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from src.core.deep import HandDrawCNN
from src.utils.loger import Logger
from sklearn import metrics
from tqdm import tqdm
import numpy as np

# split dataset func
def split_dataset(dataset):
    # Size of train, val set
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    # split data
    train_data, val_data = random_split(dataset, [train_size, val_size])
    return train_data, val_data


# evaluation func
def evaluation(labels, predicts, list_metrics):
    predicts = np.argmax(predicts, -1)
    output = {}

    if 'accuracy' in list_metrics:
        output['accuracy'] = f"{metrics.accuracy_score(labels, predicts):.2f}"
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = metrics.classification_report(labels, predicts)
    if 'report' in list_metrics:
        output['report'] = metrics.classification_report(labels, predicts)
    return output

if __name__ == "__main__":
    # define logger
    logger = Logger()
    # model arguments
    args = get_arguments()
    # dataset
    qd_dataset = HandDrawDataset(root=args.root)
    # train, val set of dataset
    train_set, val_set = split_dataset(qd_dataset)
    # train dataloader
    train_dataloader = DataLoader(train_set, args.batch_size, True, num_workers=2)
    # validation dataloader
    val_dataloader = DataLoader(val_set, args.batch_size, False, num_workers=2)
    model = HandDrawCNN(num_classes=len(CLASSES)).to(DEVICE["kernel"])

    # loss function
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.mt) if args.optim == "sgd" else Adam(
        model.parameters())

    # training phase
    print("Device: {}".format(DEVICE["name"]))
    model.train()
    best_val_loss = 0  # best validation loss
    best_val_epoch = 0  # epoch has best of loss
    for epoch in range(args.epochs):
        progress_bar = tqdm(train_dataloader, desc=f'Train epoch {epoch + 1}/{args.epochs}')
        for bid, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(DEVICE["kernel"]), labels.to(DEVICE["kernel"])
            optimizer.zero_grad()
            train_outputs = model(images)
            loss = criterion(train_outputs, labels)
            loss.backward()
            optimizer.step()
            acr = evaluation(labels.cpu().numpy(), train_outputs.cpu().detach().numpy(), ["accuracy"])["accuracy"]
            progress_bar.set_postfix({"loss": loss.item(), "accuracy": acr})
            logger.log(epoch + 1, args.epochs, f"{bid}/{len(train_dataloader)}", args.batch_size, optimizer, loss, acr)

        # validation phase
        model.eval()
        val_label_ls = []  # label list
        val_loss_ls = []  # loss list
        val_prd_ls = []  # prediction list
        for _, (val_image, val_label) in enumerate(val_dataloader):
            val_image, val_label = val_image.to(DEVICE["kernel"]), val_label.to(DEVICE["kernel"])
            with no_grad():
                predicts = model(val_image)
            batch_loss = criterion(predicts, val_label)
            val_loss_ls.append(batch_loss)
            val_prd_ls.append(predicts.clone().cpu())
            val_label_ls.extend(val_label.clone().cpu())
        # convert to numpy
        val_label_ls = np.array(val_label_ls)
        # concat item
        val_prd_ls = cat(val_prd_ls, 0)
        loss = sum(val_loss_ls) / len(val_dataloader)
        # get evaluation metrics
        evals_mtr = evaluation(val_label_ls, val_prd_ls, ["accuracy", "report"])
        print(f"Val epoch: {epoch + 1}/{args.epochs} - Loss: {loss:.2f} - Accuracy: {evals_mtr["accuracy"]}\n"
              f"{evals_mtr["report"]}", sep="\n")
        if best_val_loss == 0:
            best_val_loss = loss
        # save best model
        if loss + args.es_min_delta < best_val_loss:
            best_val_loss = loss
            best_val_epoch = epoch
            save(model.state_dict(), args.save_path)
        # (early stopping)
        # wait for es_patience to pass and if loss still does not converge then stop
        if epoch - best_val_epoch > args.es_patience > 0:
            print(f"Early stopping! Epoch: {epoch + 1} - Loss: {loss}")
            break
