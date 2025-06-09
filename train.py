from src.core.constants import DEVICE, CLASSES, SAVE_FOLDER
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from src.core.dataset import HandDrawDataset
from src.utils.utils import get_arguments
from torch import save, no_grad, load
from torch.nn import CrossEntropyLoss
from src.core.deep import HandDrawCNN
from torch.optim import Adam, SGD
from src.utils.utils import *
from tqdm import tqdm

class Trainer:
    def __init__(self, args):
        self.args = args
        self.writer = SummaryWriter("logs")

        # setup model and training components
        self.model, self.criterion, self.optimizer, self.start_epoch, self.best_loss = self.__init_components__()

    def __init_components__(self):
        model = HandDrawCNN(num_classes=len(list(CLASSES.keys()))).to(DEVICE).to(DEVICE)
        criterion = CrossEntropyLoss()
        checkpoint_best_loss = 0
        checkpoint_epoch = 0
        if self.args.optim == "sgd":
            optimizer = SGD(model.parameters(), lr=self.args.lr, momentum=self.args.mt)
        else:
            optimizer = Adam(model.parameters(), lr=self.args.lr)
        # load checkpoint
        if self.args.checkpoint:
            checkpoint = load(f"models/last_hand_draw_model.pth")
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            checkpoint_best_loss = checkpoint['best_loss']
            checkpoint_epoch = checkpoint['epoch']
        return model, criterion, optimizer, checkpoint_epoch, checkpoint_best_loss

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        running_loss = correct = total = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.args.epochs}')
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # backward
            loss.backward()
            self.optimizer.step()

            # calculate metrics
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

            # update progressbar
            progress_bar.set_postfix({"loss": loss.item(), "accuracy": correct / total})
        return running_loss / total, correct / total

    def validate(self, val_loader):
        self.model.eval()
        all_predicts = all_labels = []
        val_loss = correct = total = 0
        for images, labels in val_loader:
            with no_grad():
                all_labels.extend(labels.numpy())
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                predicts = outputs.argmax(dim=1)
                loss = self.criterion(outputs, labels)

                # calculate metrics
                val_loss += loss.item() * labels.size(0)
                correct += (predicts == labels).sum().item()
                total += labels.size(0)
                all_predicts.extend(predicts.cpu().numpy())
        return (val_loss / total, correct / total, all_labels, all_predicts)

    def train(self, train_loader, val_loader):
        # early stopping counter
        es_counter = 0
        f_mtrs = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "all_labels": [],
            "all_preds": []
        }
        for epoch in range(self.start_epoch, self.args.epochs):
            # training phase
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            f_mtrs["loss"].append(train_loss)
            f_mtrs["accuracy"].append(train_acc)

            # validation phase
            val_loss, val_acc, all_labels, all_predicts = self.validate(val_loader)
            f_mtrs["val_loss"].append(val_loss)
            f_mtrs["val_accuracy"].append(val_acc)
            f_mtrs["all_labels"] = all_labels
            f_mtrs["all_preds"] = all_predicts

            # logging
            self.log_metrics(epoch, train_loss, train_acc, val_loss, val_acc)

            # save last and best model and early stopping
            is_best = (val_loss + self.args.es_min_delta < self.best_loss) or self.best_loss == 0
            if is_best:
                self.best_loss = val_loss
                es_counter = 0
            else: es_counter += 1
            self.save_checkpoint(epoch, is_best)
            # early stopping
            if (self.args.es_patience != 0 and self.args.es_min_delta != 0) and es_counter >= self.args.es_patience:
                print(f"Early stopping! Epoch: {epoch + 1} - Loss: {val_loss}")
                break
            
            # confusion matrix plotting
            plot_confusion_matrix_tensorboard(self.writer, f_mtrs["all_labels"], f_mtrs["all_preds"], epoch)

        # final metrics plotting
        plot_metrisc_tensorboard(self.writer, f_mtrs)

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "best_loss": self.best_loss,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        # save last model
        save(checkpoint, f"{SAVE_FOLDER}/last_hand_draw_model.pth")

        # save best model
        if is_best: save(checkpoint, self.args.save_path)

    def log_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc):
        print(
            f"Epoch {epoch + 1}/{self.args.epochs} || "
            f"train_loss: {train_loss:.2f} - train_acc: {train_acc:.2f} - "
            f"val_loss: {val_loss:.2f} - val_acc: {val_acc:.2f}"
        )

        # Tensorboard logging
        self.writer.add_scalar("Train/Loss", train_loss, epoch)
        self.writer.add_scalar("Train/Accuracy", train_acc, epoch)
        self.writer.add_scalar("Validation/Loss", val_loss, epoch)
        self.writer.add_scalar("Validation/Accuracy", val_acc, epoch)

# split dataset func
def load_data(dataset):
    # Size of train, val set
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    # split data
    train_set, val_set = random_split(dataset, [train_size, val_size])
    # train dataloader
    train_loader = DataLoader(train_set, args.batch_size, True, num_workers=2)
    
    # validation dataloader
    val_loader = DataLoader(val_set, args.batch_size, False, num_workers=2)
    
    return train_loader, val_loader

if __name__ == "__main__":
    # model arguments
    args = get_arguments()
    # dataset
    qd_dataset = HandDrawDataset()
    # train, val set of dataset
    train_loader, val_loader = load_data(qd_dataset)
    
    trainer = Trainer(args)
    trainer.train(train_loader, val_loader)
