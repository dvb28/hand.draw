from ..core.constants import EPOCHS, BATCH_SIZE, SAVE_PATH, OPTIM
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=float, default=BATCH_SIZE)
    parser.add_argument("--optim", type=str, choices=["sgd", "adam"], default=OPTIM["alg"])
    parser.add_argument("--mt", type=float, default=OPTIM["momentum"])
    parser.add_argument("--lr", type=float, default=OPTIM["lr"])
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--log_path", type=str, default="logs")
    parser.add_argument("--es_min_delta", type=str, default=0,)
    parser.add_argument("--es_patience", type=int, default=4)
    parser.add_argument("--save_path", type=str, default=SAVE_PATH)
    args = parser.parse_args()
    return args